# Audio Texture Synthesis
The overall goal of this project is to generate audio textures using conventional techniques found in image texture synthesis, image in-painting and style transfer. 
This idea of texture synthesis in sound is made possible through the use of Spectrogram visualizations in which it is possible to implement a Convolutional 
Neural Network across points. This method has been proposed by researchers at Sorbonne Université and has been explored by others through the use of RI Spectrograms 
and Neural Style Transfer to name a couple of reimplementations. This project will only focus on the basic texture synthesis and aims to use all native Pytorch libraries.  
  
# Usage
Demo of network including full output can be found in notebook.py and further details about background, network architecture, results, and analysis can be found in Project_Report___ECE_176.pdf.  
  
# Future Improvements
Future improvements could be made within the network architecture itself as computational limitations forced this project to remain at only 1/4 of the number of layers
used in other projects regarding this topic. Additionally, the choice of Spectrogram could improve the networks generalization to other forms of audio to improve rythm 
detections as well as sound quality.  
  
Within this project directly, limiting the TorchAudio Spectrograms range of frequency may prove useful in audio quality improvement as well as pattern detection.
