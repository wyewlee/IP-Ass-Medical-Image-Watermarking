# Medical Image Watermarking (Invisible Watermarking)

### **A Comparative Study on Medical Image Watermarking using Hybrid Approach and RivaGAN**

### ICDXA2021: The 2nd International Conference on Digital Transformation and Applications 

*Yew Lee Wong, Jia Cheng Loh ,Chen Zhen Li, Chi Wee Tan *

*Faculty of Computing and Information Technology, Tunku Abdul Rahman University
College, Malaysia*

*Corresponding author: wongyewlee-wm19@student.tarc.edu.my*

With the increased use of electronic medical records and computer networks, Medical Image
Watermarking (MIW) now plays a very important role to preserve integrity and completeness
of medical images. As of now, there are no perfect algorithms or solutions for invisible
watermarking as there are trade-offs between visibility and robustness. In this study, we
explored multiple implementations of image watermarking techniques using Hybrid-Approach
and Deep-Learning-Approach. The experiments to measure the limitations and robustness were
done on a dataset of breast ultrasound images. 18 attacking methods were performed on the
encoded images and performance were evaluated using PSNR and NCC. Encoded images were
then being transmitted digitally using multiple transmission method to test its robustness against
transmission platform. In conclusion, the Deep-Learning Approach of RivaGAN has shown
best robustness despite many extreme attacks while the Hybrid Approach of DWT-DCT-SVD
shown the best performance in terms of imperceptibility. We reject RivaGAN as the best
solution for Medical Image Watermarking despite its robustness as it was created specifically
for video invisible watermarking.

`pip install year4`

`from year4 import cry`

