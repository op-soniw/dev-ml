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

