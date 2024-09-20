<h1 align='center'>Tennis Tracking 🎾</h1>
<p align='center'><i>Fork of <a href="https://github.com/ArtLabss/tennis-tracking">ArtLabss/tennis-tracking</a></i></p>
<p align='center'>
  <img src="https://img.shields.io/github/forks/ByAgenT/tennis-tracking.svg">
  <img src="https://img.shields.io/github/stars/ByAgenT/tennis-tracking.svg">
  <img src="https://img.shields.io/github/watchers/ByAgenT/tennis-tracking.svg">
  
  <br>
  
  <img src="https://img.shields.io/github/last-commit/ByAgenT/tennis-tracking.svg">
  <img src="https://img.shields.io/badge/license-Unlicense-blue.svg">
  <img src="https://hits.sh/github.com/ByAgenT/tennis-tracking.svg"/>
  <br>
  
</p>

<!-- 
![Forks](https://img.shields.io/github/forks/ByAgenT/tennis-tracking.svg)
![Stars](https://img.shields.io/github/stars/ByAgenT/tennis-tracking.svg)
![Watchers](https://img.shields.io/github/watchers/ByAgenT/tennis-tracking.svg)
![Last Commit](https://img.shields.io/github/last-commit/ByAgenT/tennis-tracking.svg)  
-->

<h3>How to run</h3>

<p>This project requires compatible <b>GPU</b> to install tensorflow, you can run it on your local machine in case you have one or use <a href='https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwissLL5-MvxAhXwlYsKHbkBDEUQFnoECAMQAw&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2F&usg=AOvVaw0eDNVclINNdlOuD-YTYiiB'>Google Colaboratory</a> with <b>Runtime Type</b> changed to <b>GPU</b>.</p>

> Input videos have to be rallies of the game and shouldn't contain any <strong>commercials, breaks or spectators</strong>.
  
<ol>
  <li>
    Clone this repository
  </li>
  
  ```git
  git clone https://github.com/ByAgenT/tennis-tracking.git
  ```
  
  <li>
    Install the requirements using pip 
  </li>
  
  ```python
  pip install -r requirements.txt
  ```
  
   <li>
    Run the following command in the command line
  </li>
  
  ```python
  python3 predict_video.py --input_video_path=VideoInput/video_input3.mp4 --output_video_path=VideoOutput/video_output.mp4 --minimap=0 --bounce=0
  ```
  
  <li>If you are using Google Colab upload all the files to Google Drive, including yolov3 weights from step <strong>2.</strong></li>
  
   <li>
    Create a Google Colaboratory Notebook in the same directory as <code>predict_video.py</code>, change Runtime Type to <strong>GPU</strong> and connect it to Google drive
  </li>
  
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
  <li>
    Change the working directory to the one where the Colab Notebook and <code>predict_video.py</code> are. In my case,
  </li>
  
  ```python
  import os 
  os.chdir('drive/MyDrive/Colab Notebooks/tennis-tracking')
  ```
  
  <li>
    Install only 2 requirements, because Colab already has the rest
  </li>
  
  ```python
  !pip install filterpy sktime
  ```
  
  <li>
    Inside the notebook run <code>predict_video.py</code>
  </li>
  
  ```
   !python3 predict_video.py --input_video_path=VideoInput/video_input3.mp4 --output_video_path=VideoOutput/video_output.mp4 --minimap=0 --bounce=0
  ```
  
  <p>After the compilation is completed, a new video will be created in <a href="/VideoOutput" target="_blank">VideoOutput folder</a> if <code>--minimap</code> was set <code>0</code>, if <code>--minimap=1</code> three videos will be created: video of the game, video of minimap and a combined video of both</p>
  <p><i>P.S. If you stumble upon an <b>error</b> or have any questions feel free to open a new <a href='https://github.com/ByAgenT/tennis-tracking/issues'>Issue</a> </i></p>
  
</ol>
 
<h3>Helpful Repositories</h3>
<ul>
  <li><a href="https://github.com/MaximeBataille/tennis_tracking">Tennis Tracking</a> @MaximeBataille</li>
  <li><a href="https://github.com/avivcaspi/TennisProject">Tennis Project</a> @avivcaspi</li>
  <li><a href="https://nol.cs.nctu.edu.tw:234/open-source/TrackNet/tree/master/Code_Python3">TrackNet</a></li>
</ul>

<h3>Contribution</h3>

<p>Help us by contributing, check out the <a href="https://github.com/ByAgenT/tennis-tracking/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>. Contributing is easy!</p>

<h3>References</h3>

- Yu-Chuan Huang, "TrackNet: Tennis Ball Tracking from Broadcast Video by Deep Learning Networks," Master Thesis, advised by Tsì-Uí İk and Guan-Hua Huang, National Chiao Tung University, Taiwan, April 2018. 

- Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih Peng, "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications," in the IEEE International Workshop of Content-Aware Video Analysis (CAVA 2019) in conjunction with the 16th IEEE International Conference on Advanced Video and Signal-based Surveillance (AVSS 2019), 18-21 September 2019, Taipei, Taiwan.

- Joseph Redmon, Ali Farhadi, "YOLOv3: An Incremental Improvement", University of Washington, https://arxiv.org/pdf/1804.02767.pdf
