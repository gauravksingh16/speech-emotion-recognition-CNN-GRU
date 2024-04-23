# Speech Emotion Recognition using Deep Learning

Speech Emotion Recognition (SER) is a vital aspect of human-computer interaction and affective computing, with applications ranging from virtual assistants to mental health assessment tools. I propose a comprehensive approach using multiple datasets: the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D), the Toronto emotional speech set (TESS), and the Surrey AudioVisual Expressed Emotion (SAVEE) dataset. Our methodology combines the power of Zero Crossing Rate (ZCR), Root Mean Square (RMS), and Mel-Frequency Cepstral Coefficients 
(MFCCs) as key features extracted from audio signals. I employed a Convolutional Neural Network (CNN) – Long Short-Term Memory (LSTM) architecture to capture both spectral and temporal information from the audio data. The CNN-LSTM model is trained and evaluated using these diverse datasets, encompassing a wide range of emotions such as happiness, sadness, anger, and neutral states. 

Through rigorous experimentation and evaluation, including cross-validation and comparative analysis with existing methods, we demonstrate the effectiveness of our approach in achieving state-of-the-art performance in speech emotion recognition. The results highlight the robustness and generalizability of our model across different datasets and emotion categories, paving the way for emotion-aware systems with enhanced human-computer interaction capabilities. 

------------------------------------------------------------------------------------

# Dataset

The combined dataset used in this research study is a fusion of four prominent datasets: the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA), the Toronto emotional speech set (TESS), and the Surrey Audio-Visual Expressed Emotion (SAVEE) dataset. This amalgamation of diverse datasets aims to enhance the richness, variability, and generalizability of the data for training and evaluating the proposed Speech Emotion Recognition (SER) model. The combined dataset exhibits a wide spectrum of emotional expressions, encompassing emotions such as happiness, sadness, anger, fear, surprise, disgust, and neutral states.\

-----------------------------------------------------------------------------------

# Model

I introduce a specialized model designed for extracting emotional cues from speech signals. My model combines Convolutional 1D layers for spatial pattern recognition with Long Short-Term Memory (LSTM) layers for capturing temporal dependencies. This fusion enables the model to understand both the content and context of speech, crucial for accurate emotion recognition.  I also incorporate techniques like dropout regularization and batch normalization to enhance model stability and performance. The final architecture includes dense layers for feature processing and a softmax layer for multi-class emotion classification. Through this approach, my model aims to effectively analyze speech data and predict emotional states with high accuracy and reliability. 

-----------------------------------------------------------------------------------

# Results

Our proposed CNN-LSTM model achieved a superior accuracy of 81.62%, surpassing the performance of a conventional LSTM model that attained an accuracy of 79%. This notable improvement in accuracy underscores the efficacy of our model in accurately recognizing emotional states from speech data. Moreover, our CNN-LSTM model demonstrated another significant advantage in terms of training time. The training process was notably expedited compared to the conventional LSTM model, showcasing the efficiency and scalability of our approach. 

Benefits of Using CNN+LSTM Model: 
The integration of Convolutional 1D layers (CNN) with Long Short-Term Memory (LSTM) layers offers several key advantages over traditional LSTM-based models: 
• Spatial and Temporal Feature Learning: The CNN layers excel in capturing spatial patterns in the input data, while the LSTM layers are adept at modeling temporal dependencies. By combining these capabilities, our model effectively captures both local and long-range features in speech signals, leading to improved emotion recognition performance. 

• Hierarchical Feature Extraction: The hierarchical feature extraction enabled by the CNN+LSTM architecture allows the model to learn complex representations of emotional cues, extracting meaningful features at multiple levels of abstraction. This hierarchical approach enhances the model's ability to discern subtle nuances in speech associated with different emotional states. 

• Robustness to Variability: The CNN+LSTM model exhibits robustness to variations in speech characteristics, including pitch, intensity, and duration, making it suitable for real-world applications where speech data may exhibit diverse acoustic properties. 

• Efficient Training and Inference: The parallel processing capabilities of CNN layers and the memory cell architecture of LSTM layers contribute to efficient training and inference, resulting in reduced computational costs and faster model convergence. 

Overall, our CNN+LSTM model not only outperformed conventional LSTM models but also offers a versatile and effective framework for speech emotion recognition tasks, leveraging the strengths of both convolutional and recurrent neural networks for enhanced performance and scalability. 
