## 1

### Dataset
- Thesis dataset: [Spotify dataset with lyrics](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics)
- Scraping for popularity, genre, release year: [Spotify API](https://developer.spotify.com/documentation/web-api/)

### 10 Thesis
**Predicting Music Popularity Using Machine Learning Algorithm and Music Metrics Available in Spotify**
- link: [https://d1wqtxts1xzle7.cloudfront.net/106494703/Predicting-Music-Popularity](https://d1wqtxts1xzle7.cloudfront.net/106494703/Predicting-Music-Popularity-libre.pdf?1697052722=&response-content-disposition=inline%3B+filename%3DPredicting_Music_Popularity_Using_Machin.pdf&Expires=1761041905&Signature=YL73V4prRozIK2Nsjomi9TJWyPk5TVZOqdY75N1SENUYOlWH-C3pFrIsVCCZVdws4QMY-d6DPpPE3hLVfAkzMH3GC~fn1AF4zxyeogFFbwf3piqscvBHqZaOVfR4aBwEKjMpwbB3aMrNefBVmMB2Y8OvNbvsef6f3XjQs60mWmkKNY3kfMIHYSmEpYh~84nj-2rISKTg2t9c~kggoxE7506nTN9BdIgo5ncbqEW9YaDi7qsX7XC0v56LTc4Graba67ppyKv5~mWTvHdb06TJr4Qnm4035OtDx2ylukPTf~1oE4GOiu-dQKktH2GmJMpYc-VP80gRhxWFuh5jZLBFPg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- dataset: kaggle and spotify API
- model: Random Forest Classifier gives the best results and accuracy which was up to 89%
- keywords: music popularity, Random Forest, K-Nearest Neighbor (KNN), Random Forest, Linear Support Vector Classifier

**Predicting Music Popularity: A Machine Learning Approach Using Spotify Data**
- link: https://www.scitepress.org/Papers/2024/133300/133300.pdf
- dataset: Spotify Songs dataset that recorded 114,000 songs
- model: Random Forest Performance of Regression R² score of 0.61
- keywords: music popularity, Random Forest, Linear Regression, Gradient Boosting Machines

**Predicting the mood of music from song lyrics using machine learning**
- link: https://arxiv.org/pdf/1611.00138
- dataset: fetch lyrics from LyricWikia and random 10000 songs from [Million Song Dataset](http://millionsongdataset.com/pages/getting-dataset/)
- model: The best performing model was a multinomial naive Bayes classifier (average ROC auc 0.75)
- keywords: lyrics to mood analysis, ROC curves

**Predicting song popularity based on Spotify’s audio features: insights from the Indonesian streaming users**
- link: https://www.tandfonline.com/doi/epdf/10.1080/23270012.2023.2239824?needAccess=true
- dataset: Spotify API + 92,755 rows and 20 columns [(Outdated kaggle link)](https://www.kaggle.com/yamaerenay/spotify-tracks-dataset-19222021)
- model: Random Forest with accuracy: 0.6974 and F1 Score: 0.6944
- keywords: audio features, popularity prediction, Random Forest

**Enhancing Music Emotion Classification with Lyrics and Audio Features**
- link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10470614
- dataset: 90,001 songs with valence, dominance, arousal, and emotion dimensions [MuSe: The Musical Sentiment Dataset](https://openhumanitiesdata.metajnl.com/articles/33/files/submission/proof/33-1-586-1-10-20210707.pdf)
- model: Random Forests with 73% accuracy
- keywords: music emotion classification, lyrics and audio features, sentiment dataset

**SpotiPred: A Machine Learning Approach Prediction of Spotify Music Popularity by Audio Features**
- link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9776765
- dataset: More than 170,000 songs were obtained from the [Spotify Web API and](https://developer.spotify.com/documentation/web-api/)
- model: The Random Forest model with hundreds of trees 95.37% accuracy
- keywords: only audio features, predicting popularity, Kmeans, Linear Regression, and Random Forest

**Emotion-based Analysis and Classification of Music Lyrics**
- link: https://estudogeral.uc.pt/bitstream/10316/31955/1/Emotion-based%20Analysis%20and%20Classification%20of%20Music%20Lyrics.pdf
- dataset: 180 song lyrics and 771 validation
- model: SVM best results (63.9% F-Measure)
- keywords: Support Vector Machines, K-Nearest Neighbors , Naïve Bayes.

**A Multimodal End-To-End Deep Learning Architecture for Music Popularity Prediction.**
- link: https://www.researchgate.net/publication/339480731_A_Multimodal_End-To-End_Deep_Learning_Architecture_for_Music_Popularity_Prediction/citations
- dataset: high-level audio features from Spotify, low-level audio features extracted from different audio representations such as Mel-spectrogram, Tonnetz, Chromagram or spectral centroids, a collection of text features directly gathered from lyrics, diverse information regarding artists such as the number of followers or his/her popularity, the popularity of each track and the genres associated to each track
- model: neural networks (not important for us)
- keywords: multimodal deep learning, audio features, lyrics, artist information

**Catching the Earworm: Understanding Streaming Music Popularity Using Machine Learning Models**
- link: https://www.e3s-conferences.org/articles/e3sconf/pdf/2021/29/e3sconf_eem2021_03024.pdf
- dataset: 130,663 tracks from Spotify API
- model: 0.831 Boosting tree, Random Forest and Neural Networks
- keywords: music popularity, Boosting tree, Random Forest, Neural Networks

**Beyond the Hook: Predicting Billboard Hot 100 Chart Inclusion with Machine Learning from Streaming, Audio Signals, and Perceptual Features**
- link: https://arxiv.org/pdf/2509.24856
- dataset: ?
- model: Random Forest and XGBoost
- keywords: Machine Learning · Digital music analytics · Music charts prediction · Spotify · cyclical encoding