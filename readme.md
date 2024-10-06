Introduction


The Swedish digital music company has found a place in our everyday parlance. Spotify's approach to social media is as appealing as their playlists. Their 'random act of kindness' surprises their customers by sharing a song or playlist based on the individual's taste. Touted by experts as the next social network, Spotify has an active presence on Facebook, Twitter, Instagram, YouTube and LinkedIn. In this project, we will be analyzing and extrapolating information from the Dataset found online for Spotify : A Swedish audio streaming and media
services provider founded in 2006. It has a user base of 74 million as of 2020, which keeps on growing.



Spotify introduced a "Spotify for Artists" panel, letting artists and managers access data on monthly listeners, geographical data, demographic information, music preferences and more. With this platform people are able to read calculated audio features of tracks to learn about its danceability, energy, valence, and more. For more advanced use cases, it is possible to read in-depth analysis data about tracks such as the segments, tatums, bars, beats, pitches, and more.





The dataset used for this project is from Kaggle where the data about the audio tracks has been collected for the past years usingthe Spotify web api. The dataset is in csv format and python libraries like NumPy and pandas were used for data handling, libraries like matplotlib and seaborn were used for data visualization. Work carried out includes exploratory data analysis, checking the distribution of the features in the dataset, histogram plot, regression plot for checking correlation, heatmap etc. Some of the features had some outliers which were found from a quantile plot. To get an idea about the track characteristics
(energy, acousticness, danceability, loudness etc) the dataset was aggregated over the years for easier handling.

The ideal scenario that plays out for an artist is their track reaching highest popularity status across all standings charts. Popularity is gained as more and more active listeners stream, like and share the track. Any artist would want their music to be
heard all over the world, primarily because of their love for the art, but also since it is mostly their primary source of income.  But each artist has a different style of creating music. In order to be successful, they need to know what makes a song popular.

In this project our aim was to analyze and extrapolate information from a Spotify dataset and in doing so, come to solid inferences about what really makes a song popular, majorly what the general people like in a song.

Track Duration:

●  The tracks released shows that various artists have been releasing more audio tracks each year.
●  In the 1990s, 106229 tracks were released whereas in the
2000s only 88881 audio tracks were released which can be
seen from the spike in count between 1980 and 2000.
●  In the year 2020, 13937 tracks were released which was
2030 more tracks released than the previous year(2019).
●  January has the most number of track releases(271776
releases over the years) mainly due to January being after
the festive and holiday season.


Audio Characteristics against popularity:


The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally
speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently.
●  Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. There is a slight positive correlation between energy and the popularity of the track. The values of the feature danceability are somewhat distributed normally.●  Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. There is a slight positive correlation between danceability and popularity of the track.
●  Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Values typically range between -60 and 0 db.There is a positive correlation between loudness and track popularity. Loudness is normally distributed with a negative skewness.
●  Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech. Values below 0.33 most likely represent music and other non-speech-like tracks.


Energy









Danceability

Loudness








Speechiness





Audio features like energy, danceability and loudness kept increasing over the years but speechiness had some increase around the 1980's and then remained stagnant.

Graph visualization of artist collaboration:


If there is a collaboration, each row in the dataset has an artist's name who has collaborated for the song. What we have tried to do was create a graph of all the collaborations between the artists.

First, we get the unique name of all artists in the dataset and the name of all the artists who have collaborated for a song by basic text cleaning and store this in a list. Then we get the edges of the graph from the artist collaboration list.

Then we further filter these edges to find the unique ones as many of the edges were occurring more than once. This makes sense as artists sometimes collaborated with another artist on more than one song. As the data is huge here it contains about 5 lakh rows, we have grouped the tracks by year and a graph will be created for a particular year.

And also, we have tried to further group the artist by the language we get from the artist's name by using a nlp library called Lang detect(Language detection algorithm is non-deterministic, so if we run it on a text which is either too short or too ambiguous like the name of a person, we get different results every time we run it).Then the languages are colour coded for easier visualisation. We used pyvis for both network creation and visualisation. The nodes and the edge in the undirected graph and the nodes are added and then given a specific colour depending on the
language detected.



Graph showing all the artist collaboration in the year 2020









Some cropped part of the above graph detailing the node’s(artist’s) neighbours and the language of the node(artist)Graph visualization of artist network with weighted edges:


In  this  section,  we  are  trying  to  visualize  the  collaboration between    various    artists    and    the    effectiveness    of    their collaborations.  The effectiveness is measured using two metrics: Average    popularity    score    between   the   collaborations   and Frequency of the collaboration. To do so, we preprocess the artist column by first transforming it to a list of two artist pairs and their respective popularity scores. If two pairs are redundant, then they are merged into one. Their counts are increased by one and their popularity  scores  are  also  added.  Then,  the  score  column  is divided  by  the  count  column.  The  count  column  also  gives  the frequency of their collaborations.


To  visualize  the  network,  pyvis  library  is  used.  The  pyvis library is meant for quick generation of visual network graphs with minimal  python  code.  It  is  designed  as  a  wrapper  around  the popular  Javascript  visJS  library.  To  visualize  the  graphs,  the  list datasets   described   above   are   converted   into   network   class objects.  Unique  artists  become  the  node  and  the  collaboration acts as edges. In the first network object, the average popularity score  becomes  the  edge  weights  and  in  the  second  one,  the frequency  of  collaboration(count)  becomes  the  edge  weights. After  creating  the  objects,  the  networks  are visualized by calling "net.show('network_name.html')".   Sample  outputs  for  both  the graphs are shown in the following figures.

Fig: Graph visualisation of artists network using frequency of collaboration as edges(20 edges).




Fig: Graph visualisation of artists network using average popularity score as edges(20 edges).Getting the important features:



To get the features which are important in deciding the track popularity, we used a lightgbm classifier to fit the data and predict the output. Originally the popularity feature has values ranging from 0-100, but for a baseline model we have grouped it into three different values(0-33,33-66,67-100).






As we can see from above the audio characteristics like song duration, accousticness, loudness, energy, etc have more importance in deciding the track popularity.




Conclusion:
What we have done is mainly focused on analysis of the data about the tracks and how artist’s are collaborating. From the audio features analysis we have inferred the importance of somecharacteristics more than others, and from the network visualization we got to know that the number of people who are collaborating have been increasing. Many collaborations are even breaking the language barrier. From the weighted graphs we can know which collaboration would yield more popularity.




References:
1.  Dataset taken from:
 https://w w w.kag gle.com/yamaerenay/spotif y-dataset-19212020-160k-track 
s
2.  Data information:
 https://developer.spotif y.com/documentation/web-api/reference/#endpoin
 t-get-audio-features