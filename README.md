# mvse - A Multimodal Video Search Engine

**_ARCHIVED._**

This repository contains a multimodal semantic search engine for video datasets. It is multimodal because it
uses not only written tags describing the video, but tries to obtain text descriptions from videos frames.
It is semantic because it attempts to improve accuracy of the search by trying to search for the meaning of
the stored data, not doing a simple lexical search. This project leverages existing open-source deep learning 
models to achieve such task. The description of the models used (and also attempted) are found below.

The aim is to understand and to improve search capabilities in video catalogs, providing more relevant and efficient
results.

## Dataset 

Conjuntos de videos podem ser encontrados [nessa página](https://github.com/xiaobai1217/Awesome-Video-Datasets) do GitHub.
Especificamente, dessa página, da parte de Multimodal Learning, eu retirei a referência para [MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](https://github.com/pliang279/MultiBench). O trabalho que explica o que é Multibench está [aqui](https://arxiv.org/abs/2107.07502).


Eu imagino que os vídeos possam vir desse repositório de GitHub.

### Papers

#### Multimodal Machine Learning
[Tutorial on Multimodal Machine Learning](https://aclanthology.org/2022.naacl-tutorials.5.pdf)

#### Multimodal Search Engines
[Context-aware Querying for Multimodal Search Engines](https://www.cs.upc.edu/~tsteiner/papers/2012/context-aware-querying-mmm2012.pdf)
[Apply Multimodal Search and Relevance Feedback In a Digital Video Library](https://web.archive.org/web/20120415041659/http://www.informedia.cs.cmu.edu/documents/zhong_thesis_may00.pdf)

### TODO: Organize by Topic - Google Searches

- https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings
- https://www.elastic.co/blog/how-to-deploy-nlp-text-embeddings-and-vector-search
- https://platform.openai.com/docs/guides/embeddings
- https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c
- https://towardsdatascience.com/beyond-ctrl-f-44f4bec892e9
- https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
- https://dev.to/mage_ai/how-to-build-a-search-engine-with-word-embeddings-56jd
- https://huggingface.co/course/chapter5/6?fw=tf
- https://catalog.workshops.aws/semantic-search/en-US/module-1-understand-internal/semantic-search-technology
- Extending Full Text Search for Legal Document Collections Using Word Embeddings
- https://www.scribd.com/document/492791276/Using-Word-Embeddings-for-Text-Search1
- https://blog.dataiku.com/semantic-search-an-overlooked-nlp-superpower
- https://simonwillison.net/2023/Jan/13/semantic-search-answers/
- https://www.buildt.ai/blog/3llmtricks
- https://www.algolia.com/blog/ai/what-is-vector-search/https://qdrant.tech/benchmarks/?gclid=CjwKCAjw__ihBhADEiwAXEazJtsrttmfhWQWIx-xZ2cATXTa2Omoc8RczL_6Bk1NnX_BmNND33xWoxoCqjAQAvD_BwE
- https://cloud.google.com/vertex-ai/docs/matching-engine/overview
- https://developer.huawei.com/consumer/en/doc/development/hiai-Guides/text-embedding-0000001055002819
- https://applyingml.com/resources/search-query-matching/
- https://betterprogramming.pub/openais-embedding-model-with-vector-database-b69014f04433
- 

#### term: embedding-based retrieval

- Modern Information Retrieval: The Concepts and Technology behind Search

- Word embedding based generalized language model for information retrieval

- Embedding-based Query Language Models

- Lbl2Vec: an embedding-based approach for unsupervised document retrieval on predefined topics

- Neural embedding-based indices for semantic search

- Embedding-based news recommendation for millions of users

- Divide and Conquer: Towards Better Embedding-based Retrieval for Recommender Systems From a Multi-task Perspective

- Embedding based on function approximation for large scale image search

- Pre-training Tasks for Embedding-based Large-scale Retrieval
- QuadrupletBERT: An Efficient Model For Embedding-Based Large-Scale
Retrieval
- A text-embedding-based approach to measuring patent-to-patent technological similarity

### Other references

Y. Bengio, A. Courville, and P. Vincent. 2013. Representation Learning: A Review and New Perspectives. IEEE Transactions on Pattern Analysis and Machine
Intelligence 35, 8 (Aug 2013), 1798–1828

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding.
CoRR abs/1810.04805 (2018). arXiv:1810.04805 http://arxiv.org/abs/1810.04805
[5] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized Product
Quantization for Approximate Nearest Neighbor Search. In The IEEE Conference
on Computer Vision and Pattern Recognition (CVPR).
[6] Alexander Hermans, Lucas Beyer, and Bastian Leibe. 2017. In Defense of
the Triplet Loss for Person Re-Identification. CoRR abs/1703.07737 (2017).
arXiv:1703.07737 http://arxiv.org/abs/1703.07737
[7] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry
Heck. 2013. Learning Deep Structured Semantic Models for Web Search Using
Clickthrough Data. In Proceedings of the 22nd ACM International Conference on
Information and Knowledge Management (CIKM ’13). Association for Computing
Machinery, New York, NY, USA, 2333–2338.
[8] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search. IEEE Trans. Pattern Anal. Mach. Intell. 33, 1 (Jan.
2011), 117–128.
[10] Yann LeCun, Yoshua Bengio, and Geoffrey E. Hinton. 2015. Deep learning. Nature
521, 7553 (2015), 436–444.
Victor Lempitsky. 2012. The Inverted Multi-Index. In Proceedings of the 2012 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR) (CVPR ’12). IEEE
Computer Society, USA, 3069–3076.
] Hang Li and Jun Xu. 2014. Semantic Matching in Search. Now Publishers Inc.,
Hanover, MA, USA.
Bhaskar Mitra and Nick Craswell. 2018. An Introduction to Neural Information
Retrieval. Foundations and Trends® in Information Retrieval 13, 1 (December
2018), 1–126.
[14] Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. FaceNet: A unified
embedding for face recognition and clustering.. In CVPR. IEEE Computer Society,
815–823. http://dblp.uni-trier.de/db/conf/cvpr/cvpr2015.html#SchroffKP15
[15] Josef Sivic and Andrew Zisserman. 2003. Video Google: A Text Retrieval Approach
to Object Matching in Videos. In Proceedings of the Ninth IEEE International
Conference on Computer Vision - Volume 2 (ICCV ’03). IEEE Computer Society,
USA, 1470.
[16] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. 2015. Deep Metric
Learning via Lifted Structured Feature Embedding. CoRR abs/1511.06452 (2015).
arXiv:1511.06452 http://arxiv.org/abs/1511.06452

[17] Chao-Yuan Wu, R. Manmatha, Alexander J. Smola, and Philipp Krähenbühl. 2017.
Sampling Matters in Deep Embedding Learning. CoRR abs/1706.07567 (2017).
arXiv:1706.07567 http://arxiv.org/abs/1706.07567

Yuhui Yuan, Kuiyuan Yang, and Chao Zhang. 2017. Hard-Aware Deeply Cascaded
Embedding. In The IEEE International Conference on Computer Vision (ICCV)


[//]: # (## Background)

[//]: # (Exact match similarity is a particularly effective approach for search queries, but only when users know the exact text )

[//]: # (they are seeking. In current search engines, this limitation is addressed with more advanced techniques, such as word )

[//]: # (embeddings to induce semantic representations.)

[//]: # (New systems leverage machine learning and natural language processing techniques to better understand the underlying )

[//]: # (meaning and context of the search queries, enabling more effective retrieval of relevant content. To find content, )

[//]: # (therefore, it makes sense to derive meaning from various modalities, including text and image. \cite{onal2018neural}\\)

[//]: # (This is also true for videos, where semantic search engines could be employed to provide efficient and relevant )

[//]: # (results by combining various modalities, such as speeches, video frames, and video descriptions.)

[//]: # (Currently, this task is achievable just by leveraging state-of-the-art deep-learning models that have been developed )

[//]: # (for embedding generation &#40;e.g., MiniLM-L6 and alternatives&#41;, frame description &#40;e.g., CLIP-based CoCa and )

[//]: # (alternatives&#41;, and vector databases &#40;e.g., Weaviate and alternatives&#41;.)

[//]: # (Such a semantic search engine could be envisioned with three different components.)


[//]: # (The first component is a pipeline which)

[//]: # (\begin{itemize})

[//]: # (    \item Selects time-frames in the video based on a ``linguistic chunk'';)

[//]: # (    \item Samples frames in the video and get the descriptions of the frames;)

[//]: # (    \item Combines both the video descriptions for each time frame and the frame descriptions to get the embedding;)

[//]: # (    \item At query inference time, gets the embedding of the query and searches for the embedding in the vector database, in order to retrieve the time-frames and the video.)

[//]: # (\end{itemize})

[//]: # (Some considerations and challenges for the first component are: &#40;a&#41; the performance and relevance of search results for videos without transcripts are still uncertain, as are the optimal combinations of what linguistic chunks to use &#40;which could be speeches or descriptions, or maybe depend on the nature of the transcription&#41;; &#40;b&#41; how to combine the video descriptions and the frame descriptions &#40;maybe they should not even be combined and two different searches should be conducted and only the results should be combined&#41;; &#40;c&#41; what vector database &#40;and also what configurations and algorithms should be used&#41;, embedding models, metrics for how to search for embeddings, and image description models should be used; &#40;d&#41; how to evaluate the performance, what parameters should be considered during and after development.)

[//]: # (The second component is a query module that)

[//]: # (\begin{itemize})

[//]: # (    \item Receives the queries as questions;)

[//]: # (    \item Processes the query text and passes it to the search engine;)

[//]: # (    \item Retrieves results as texts, clips, videos, and timestamps, identifying exactly the most relevant results in each video;)

[//]: # (    \item Builds the answers with the question and answering model.)

[//]: # (\end{itemize})

[//]: # (Some considerations and challenges for the second component are: &#40;a&#41; how the queries will be received and how this query should be processed for database search; &#40;b&#41; what models should be used for this part, the question/answering generation will be done with Hugging Face's flan-t5-xxl or a similar model; &#40;c&#41; how services such as Haystack, Langchain, and IndexGPT could play a role here.\\)

[//]: # (The third component is the question/answer query platform, which is a website that communicates directly with the other components through a RESTful API and may contain a ``chat'' bar.)

[//]: # (I cannot think of real challenges for the third component.)

[//]: # (\section*{Goal and Objectives})

[//]: # (\label{sec:goals})

[//]: # (\textbf{Goal:} The primary goal of this research is to develop a comprehensive multimodal semantic search engine for video datasets that effectively combines text and video modalities to provide accurate, efficient, and relevant search results. This could be achieved with a 175h project.\\)

[//]: # (To achieve this goal, the project will focus on the following objectives:\\)

[//]: # (\textbf{Objective 1:} Develop a pipeline for generating and combining embeddings from time-aligned transcripts and video frame descriptions.)

[//]: # (\begin{itemize})

[//]: # (\item Determine the optimal method for selecting linguistic chunks from video transcripts.)

[//]: # (\item Evaluate and select the most suitable image description models for generating video frame descriptions.)

[//]: # (\item Establish an effective approach for combining video descriptions with frame descriptions to generate a comprehensive embedding, as well as evaluate and select the most suitable embedding generation model and define a metric for a pair of embeddings.)

[//]: # (\item Evaluate and select the most suitable vector database to store the embedding.)

[//]: # (\item Establish evaluation metrics and methodologies for assessing the relevance and performance of the search results, and, with this, optimize the choice of vector databases, embedding models, and image description models based on the evaluation results.)

[//]: # (\end{itemize})

[//]: # (\textbf{Objective 2:} Design and implement a query module for processing user queries and generating relevant search results.)

[//]: # (\begin{itemize})

[//]: # (\item Evaluate and select the most appropriate question-answering model &#40;e.g., flan-t5-xxl or similar&#41; for generating answers based on the available computing resources.)

[//]: # (\item Assess the viability of integrating services such as Haystack, Langchain, and IndexGPT into the query module.)

[//]: # (\item Develop a system to receive, process, and query user questions in the database.)

[//]: # (\end{itemize})

[//]: # (\textbf{Objective 3:} Develop a user-friendly query platform that communicates with the other components through a RESTful API.)

[//]: # (\begin{itemize})

[//]: # (\item Design a web-based interface featuring a chat field to facilitate user interaction with the search engine.)

[//]: # (\item Implement a seamless integration of the query platform with the pipeline and query module using a RESTful API.)

[//]: # (\item Ensure the query platform provides an intuitive and accessible user experience, enabling users to efficiently explore large video catalogs and discover relevant content.)

[//]: # (\end{itemize})

[//]: # (\textbf{Objective 4:} Deploy the whole system and create methods to continuously refine and iterate on the system based on user feedback and evolving requirements.)

[//]: # (\begin{itemize})

[//]: # (\item Deploy the system using Red Hen Lab's infrastructure, according to its guidelines.)

[//]: # (\item Develop methods to gather user feedback on the search engine's performance, interface, and overall user experience.)

[//]: # (\item Identify areas for improvement and implement necessary changes to the pipeline, query module, and query platform.)

[//]: # (\end{itemize})

[//]: # (By addressing these objectives, the project will contribute to the development of a cutting-edge multimodal semantic search engine that enables users to effectively browse and search large video catalogs, delivering relevant and efficient results that meet their diverse information needs.)

[//]: # (\section*{Methods})


[//]: # (The challenges this project will tackle include developing a multimodal semantic search engine for videos with time-aligned transcripts, evaluating various solutions for vector databases, embedding models, and image description models, and determining the best combination of components to use. The evaluation process will be extensive, and each component will be studied cautiously to ensure optimal performance.\\)

[//]: # (The chosen method involves leveraging existing open-source deep learning models, such as CLIP for image description, MiniLM-L6 for embedding generation, and Weaviate for vector database management. This approach is chosen because it harnesses state-of-the-art technologies and allows for efficient and modular development of the search engine.\\)

[//]: # (The result of this project will be a functional video search platform that allows users to query and retrieve videos based on questions and answer. The output will consist of a list of video results ranked by their relevance to the query, with the possibility of further refining or filtering the results.\\)

[//]: # (In the future, this project can be improved and extended by incorporating more advanced machine learning techniques for better understanding and indexing video content with and without transcripts. This could involve exploring different embedding models, and refining the pipeline to enhance search efficiency and relevance. Furthermore, collaboration with other researchers and developers can lead to the creation of more sophisticated multimodal search engines that cater to a broader range of user needs.\\)

[//]: # (I am highly motivated to continue with this research, as I find it to be very relevant. I will be happy to help others build on top of the project and share my knowledge and experience.\\)
