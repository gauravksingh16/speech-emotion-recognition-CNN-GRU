# Speech Emotion Recognition using Deep Learning

Speech Emotion Recognition (SER) stands as a cornerstone in human-computer interaction and affective computing, finding applications from virtual assistants to mental health assessment tools. To address this, I propose a comprehensive methodology leveraging multiple datasets: the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D), the Toronto emotional speech set (TESS), and the Surrey AudioVisual Expressed Emotion (SAVEE) dataset. Our approach integrates Zero Crossing Rate (ZCR), Root Mean Square (RMS), and Mel-Frequency Cepstral Coefficients (MFCCs) as critical features extracted from audio signals.

In advancing our methodology, we adopt a Convolutional Neural Network (CNN) architecture combined with a Gated Recurrent Unit (GRU) layer, enhanced by an attention mechanism. This amalgamation allows us to capture emotional speech data's spectral intricacies and temporal dynamics.

Our proposed CNN+GRU with attention model undergoes rigorous training and evaluation across diverse datasets, encompassing a spectrum of emotions, including happiness, sadness, anger, and neutral states. Through cross-validation and comparative analyses against existing methods, we showcase the efficacy of our approach in achieving state-of-the-art performance in speech emotion recognition.

The results underscore the robustness and generalizability of our model across varied datasets and emotion categories, paving the way for emotion-aware systems with heightened human-computer interaction capabilities and enhanced user experiences.

------------------------------------------------------------------------------------

# Dataset

The combined dataset used in this research study is a fusion of four prominent datasets: the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA), the Toronto emotional speech set (TESS), and the Surrey Audio-Visual Expressed Emotion (SAVEE) dataset. This amalgamation of diverse datasets aims to enhance the data's richness, variability, and generalizability for training and evaluating the proposed Speech Emotion Recognition (SER) model. The combined dataset exhibits a wide spectrum of emotional expressions, encompassing happiness, sadness, anger, fear, surprise, disgust, and neutral states.

-----------------------------------------------------------------------------------

# Model

I introduce a specialized model meticulously crafted for extracting emotional cues from speech signals. This model harmoniously merges Convolutional 1D layers for spatial pattern recognition with Gated Recurrent Unit (GRU) layers for capturing temporal dependencies, augmented by an Attention Mechanism for enhanced context understanding. This fusion empowers the model to decipher the content and context of speech, a pivotal aspect for accurate emotion recognition.

To bolster stability and performance, I integrate techniques like dropout regularization and batch normalization into the model architecture. These techniques are crucial in mitigating overfitting and ensuring robust performance across diverse datasets.

The finalized architecture encompasses dense layers tailored for comprehensive feature processing, culminating in a softmax layer for precise multi-class emotion classification. Through this sophisticated approach, my model aims to adeptly analyze speech data and predict emotional states with unparalleled accuracy and reliability.

-----------------------------------------------------------------------------------

# Results

My proposed CNN+GRU model demonstrated a remarkable accuracy of 83.54%, showcasing a substantial improvement over the conventional LSTM model. This notable enhancement in accuracy reaffirms the effectiveness of our model in accurately discerning emotional states from speech data. Additionally, our CNN+GRU model exhibited another significant advantage in terms of training time, further emphasizing the efficiency and scalability of our approach. 

Benefits of Using CNN+GRU Model: 
The integration of Convolutional 1D layers (CNN) with Gated Recurrent Unit (GRU) layers offers several key advantages over traditional LSTM-based models:

• Spatial and Temporal Feature Learning: The CNN layers excel in capturing spatial patterns in the input data, while the GRU layers are proficient at modeling temporal dependencies. This combination allows our model to effectively capture both local and long-range features in speech signals, leading to superior emotion recognition performance.

• Hierarchical Feature Extraction: The CNN+GRU architecture facilitates hierarchical feature extraction, enabling the model to learn complex representations of emotional cues at multiple levels of abstraction. This hierarchical approach enhances the model's ability to discern subtle nuances in speech associated with different emotional states.

• Robustness to Variability: The CNN+GRU model demonstrates robustness to variations in speech characteristics such as pitch, intensity, and duration, making it suitable for real-world applications with diverse acoustic properties.

• Efficient Training and Inference: Leveraging the parallel processing capabilities of CNN layers and the gating mechanism of GRU layers contributes to efficient training and inference, resulting in reduced computational costs and faster model convergence.

In summary, our CNN+GRU model surpassed conventional LSTM models in accuracy and offers a versatile and effective framework for speech emotion recognition tasks. By leveraging the strengths of both convolutional and recurrent neural networks, our model achieves enhanced performance, scalability, and efficiency in analyzing speech data for emotion recognition.
