# NotoSansCJKkr-Medium.otf 파일 업로드 

폰트 파일을 업로드 한다

# 파일카피 
```
!mv NotoSansCJKkr-Medium.otf /usr/share/fonts/truetype/
```

### 팁 

##### 오류
import warnings
warnings.filterwarnings('ignore')

##### 그래프에 retina display 적용

```
%config InlineBackend.figure_format = 'retina'
```

##### 나눔고딕 설치

```
!apt -qq -y install fonts-nanum > /dev/null
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
```

#### 기본 글꼴 변경

```
import matplotlib as mpl
mpl.font_manager._rebuild()
mpl.pyplot.rc('font', family='NanumBarunGothic')
```

#### 한글폰트 사용하기

* 한글이 깨져보이는 것을 해결하기 위해 한글폰트를 사용해야 한다.
* 여기에서는 나눔바른고딕을 사용하도록 한다. 
    * 이때 폰트가 로컬 컴퓨터에 설치되어 있어야한다. 
    * 나눔고딕은 무료로 사용할 수 있는 폰트다. 
    * 참고 : [네이버 나눔글꼴 라이선스](https://help.naver.com/support/contents/contents.nhn?serviceNo=1074&categoryNo=3497)
* 한글을 사용하기 위해서는 ggplot에서 theme에 폰트를 지정해 주면된다.
* 아래의 문서를 참고하면 **element_text**와 관련된 옵션을 볼 수 있다.
* 참고 : [plotnine.themes.element_text — plotnine 0.3.0 documentation](http://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.element_text.html)


#### 글씨가 겹쳐보이지 않도록 rotation 추가

```
(ggplot(petitions)
 + aes('category') 
 # XY 축 변경 >> + aes(x='category', y='votes') 
 # 서클 포인트 + aes(x='category', y='votes', fill='answer')
 # 서클 포인트 + geom_point()
 + geom_bar(fill='green')
 + theme(text=element_text(family='NanumBarunGothic'),
        axis_text_x=element_text(rotation=60))
)
```

#### 단어수 

```
# 단어 수
%time petitions['title_num_words'] = petitions['title'].apply(lambda x: len(str(x).split()))
# 중복을 제거한 단어 수
%time petitions['title_num_uniq_words'] = petitions['title'].apply(lambda x: len(set(str(x).split())))
```

#### 특정 단어가 들어가는 단어만

```
import re
p = r'.*(돌봄|아이|초등|보육).*'
care = petitions[petitions['title'].str.match(p) |
           petitions['content'].str.match(p, flags=re.MULTILINE)]
care.shape
```
