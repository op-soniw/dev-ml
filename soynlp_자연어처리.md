***본글은 [국민청원으로 파이썬 자연어처리 입문하기](https://www.youtube.com/playlist?list=PLaTc2c6yEwmrtV81ehjOI0Y8Y-HR6GN78)의 내용을 기초로 한 것입니다.***

# 자연어처리 시작하기

```
!pip install soynlp
#워드 클라우드 
!pip install wordcloud
```

```
!pip show soynlp
```

### 토큰화

```
from soynlp.tokenizer import RegexTokenizer
tokenizer = RegexTokenizer()

```


### 개행문자 제거

```
def preprocessing(text):
    # 개행문자 제거
    text = re.sub('\\\\n', ' ', text)
    return text
```

## 워드 클라우드 찍기 

```
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline

def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(
                        font_path = fontpath, 
                        stopwords = stopwords_kr, 
                        background_color = backgroundcolor, 
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 
    
```

### STOPWORD 처리 

명사 추출

```
from soynlp.noun import LRNounExtractor
```

불용어 처리 

```
%%time
noun_extractor = LRNounExtractor(verbose=True)

noun_extractor.train(sentences)
nouns = noun_extractor.extract()

추출된 명사를 찍어봅니다.
%time displayWordCloud(' '.join(nouns))
```


