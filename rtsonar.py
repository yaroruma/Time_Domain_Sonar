# Import functions and libraries
from __future__ import division
import numpy as np
import matplotlib.cm as cm
from scipy import signal
from scipy import interpolate
from numpy import *
import threading,time, Queue, pyaudio
import bokeh.plotting as bk
from bokeh.models import GlyphRenderer
from bokeh.io import push_notebook
from IPython.display import clear_output

bk.output_notebook()


def put_data( Qout, ptrain, Twait, stop_flag):
    while( not stop_flag.is_set() ):
        if ( Qout.qsize() < 2 ):
            Qout.put( ptrain )
            
        time.sleep(Twait)
            
    Qout.put("EOT")
            
def play_audio( Qout, p, fs, stop_flag, dev=None):
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while ( not stop_flag.is_set()):
        data = Qout.get()
        if data is "EOT" :
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
        except:
            break
    ostream.stop_stream();
    ostream.close()
            
def record_audio( Qin, p, fs, stop_flag, dev=None,chunk=1024):
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (  not stop_flag.is_set() ):
        try:  # when the pyaudio object is destroyed, stops
            data_str = istream.read(chunk) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        Qin.put( data_flt ) # append to list
    istream.stop_stream();
    istream.close()
    Qin.put("EOT")

    
def signal_process( Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag  ):
    # Signal processing function for real-time sonar
    # Takes in streaming data from Qin and process them using the functions defined above
    # Uses the first 2 pulses to calculate for delay
    # Then for each Nseg segments calculate the cross correlation (uses overlap-and-add)
    # Inputs:
    # Qin - input queue with chunks of audio data
    # Qdata - output queue with processed data
    # pulse_a - analytic function of pulse
    # Nseg - length between pulses
    # Nplot - number of samples for plotting
    # fs - sampling frequency
    # maxdist - maximum distance
    # temperature - room temperature

    crossCorr = functions[2]
    findDelay = functions[3]
    dist2time = functions[4]
    
    # initialize Xrcv 
    Xrcv = zeros( 3 * Nseg, dtype='complex' );
    cur_idx = 0; # keeps track of current index
    found_delay = False;
    maxsamp = min(int(dist2time( maxdist, temperature) * fs), Nseg); # maximum samples corresponding to maximum distance
    
    while(  not stop_flag.is_set() ):
        
        # Get streaming chunk
        chunk = Qin.get();
        if (chunk is "EOT"):
            break;
        Xchunk =  crossCorr( chunk, pulse_a ) 
        
        # Overlap-and-add
        Xrcv[cur_idx:(cur_idx+len(chunk)+len(pulse_a)-1)] += Xchunk;
        cur_idx += len(chunk)
        
        if( found_delay and (cur_idx >= Nseg) ):
            
            # crop a segment from Xrcv and interpolate to Nplot
            Xrcv_seg = (abs(Xrcv[:maxsamp].copy()) / abs( Xrcv[0] )) ** 0.5 ;
            interp = interpolate.interp1d(r_[:maxsamp], Xrcv_seg)
            Xrcv_seg = interp( r_[:maxsamp-1:(Nplot*1j)] )
            
            # remove segment from Xrcv
            Xrcv = np.roll(Xrcv, -Nseg );
            Xrcv[-Nseg:] = 0
            cur_idx = cur_idx - Nseg;
            
            Qdata.put( Xrcv_seg );
            
        elif( cur_idx > 2 * Nseg ):
            
            # Uses two pulses to calculate delay
            idx = findDelay( abs(Xrcv), Nseg );
            
            Xrcv = np.roll(Xrcv, -idx );
            Xrcv[-idx:] = 0;
            cur_idx = cur_idx - idx;
            found_delay = True
             
    Qdata.put("EOT")
            
            
def image_update( Qdata, fig, Nrep, Nplot, stop_flag):
    renderer = fig.select(dict(name='echos', type=GlyphRenderer))
    source = renderer[0].data_source
    img = source.data['image'][0];
    
    while(  not stop_flag.is_set() ):
        new_line = Qdata.get();
        
        if new_line is "EOT" :
            break
    
        img = np.roll( img, 1, 0);
        view = img.view(dtype=np.uint8).reshape((Nrep, Nplot, 4))
        view[0,:,:] = cm.jet(new_line) * 255;
    
        source.data['image'] = [img]
        push_notebook()
        Qdata.queue.clear();
        
    

        
def rtsonar( f0, f1, fs, Npulse, Nseg, Nrep, Nplot, maxdist, temperature, functions ):

    clear_output();
    genChirpPulse = functions[0]
    genPulseTrain = functions[1]
    
    pulse_a = genChirpPulse(Npulse, f0,f1,fs) * np.hanning(Npulse)
    pulse = np.real(pulse_a)
    ptrain = genPulseTrain(pulse, Nrep, Nseg)
    
    # create an input output FIFO queues
    Qin = Queue.Queue()
    Qout = Queue.Queue()
    Qdata = Queue.Queue()

    # create a pyaudio object
    p = pyaudio.PyAudio()
    
    # create black image
    img = np.zeros((Nrep,Nplot), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((Nrep, Nplot, 4))
    view[:,:,3] = 255;
    
    # initialize plot
    fig = bk.figure(title = 'Sonar',  y_axis_label = "Time [s]", x_axis_label = "Distance [cm]",
                    x_range=(0, maxdist), y_range=(0, Nrep * Nseg / fs ) , 
                    plot_height = 400, plot_width = 800 )
    fig.image_rgba( image = [ img ], x=[0], y=[0], dw=[maxdist], dh=[Nrep * Nseg / fs ], name = 'echos' )
    bk.show(fig, notebook_handle=True)

    # initialize stop_flag
    stop_flag = threading.Event()

    # initialize threads
    t_put_data = threading.Thread(target = put_data,   args = (Qout, ptrain, Nseg / fs, stop_flag  ))
    t_rec = threading.Thread(target = record_audio,   args = (Qin, p, fs, stop_flag  ))
    t_play_audio = threading.Thread(target = play_audio,   args = (Qout, p, fs, stop_flag  ))
    t_signal_process = threading.Thread(target = signal_process, args = ( Qin, Qdata, pulse_a, Nseg, Nplot, fs, maxdist, temperature, functions, stop_flag))
    t_image_update = threading.Thread(target = image_update, args = (Qdata, fig, Nrep, Nplot, stop_flag ) )

    # start threads
    t_put_data.start()
    t_rec.start()
    t_play_audio.start()
    t_signal_process.start()
    t_image_update.start()

    return stop_flag
    

