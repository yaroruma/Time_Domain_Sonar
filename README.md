# Time Domain Realtime Sonar

## Brief
디지털 신호 처리 과목 기말 프로젝트 코드입니다. 노트북의 스피커와 마이크를 이용해 주변의 물체를 탐지하는 소나를 구현했습니다.

## Mission
chirp 신호로 pulse train을 스피커로 출력, 반사된 소리를 cross-correlation하여 주변 물체와의 거리를 도출하는 함수 구현.

## Environment
 - anaconda 2
 - jupyter notebook
 - python 2.7

## Requirements
 - pyaudio 0.2.7
 - scipy
 - bokeh
 - numpy
 - matplotlib

## Usage
1. Open `lab2-TimeDomain-RealTime-Sonar.ipynb` with jupyter notebook.
2. Run 3rd cell to start sonar.
3. Move things around speakers and watch the difference.
4. Run 4th cell to stop sonar.

실제 DSP를 통해 접근하는 과정은 `lab1-TimeDomain-Sonar.ipynb`에 나와있습니다.   
프로젝트는 UC Berkeley의 EE-123 수업의 1번째 과제와 동일합니다.
