Data Augmentation
======================

# 1. 본 프로젝트에 대하여

## 1.1 Data Augmentation 방법
BART, T5를 이용해 IMDB 리뷰 데이터 등을 요약하고, 그 요약된 문서를 확장 데이터로 사용하는 방식.
요약 방법으로는 Abstractive Summarization을 이용함.

## 1.2 검증 방법
분류모델로서 CNN, 양방향 GRU + 바다나우 어텐션, BERT를 이용해 검증.

학습, 테스트, 검증 데이터의 비율은 8:1:1로 함.

데이터 확장은 절대적인 데이터의 수가 적을수록 효과 상승이 극대화되기 때문에 학습 데이터의 수를 여러개로 나눠서 검증.

IMDB 기준, 1%인 500개부터 100%인 50,000개까지 나눠서.

성능의 척도는 accuracy로 판단. 

1번만 해서는 결과가 정확하지 않으니 CNN과 GRU는 10번, BERT는 5번을 테스트한 평균치로 계산.

### 1.2.1 CNN을 사용하는 이유
문장의 국부적인 정보를 보존함으로써 데이터 확장의 성능을 효과적으로 평가할 수 있다고 판단되기 때문.

Yoon Kim. (2014). Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.

### 1.2.2 GRU를 이용하는 이유
데이터의 양이 적을때 LSTM보다 GRU의 성능이 더 좋다는 결과가 있기 때문.
기본적으로 데이터 확장은 데이터의 양이 적을때 사용되기 때문에 LSTM보다 GRU가 성능 확인에 적합할 것이라 생각.
더 정확한 결과를 위해 양방향, 바다나우 어텐션 추가. 

Yang, S., Yu, X., & Zhou, Y. (2020, June). Lstm and gru neural network performance comparison study:  Taking  yelp  review  dataset  as  an  example.  In  2020  International  workshop  on  electronic communication and artificial intelligence (IWECAI) (pp. 98-101). IEEE.

### 1.2.3 BERT를 이용하는 이유
지금까지 대부분의 데이터 확장 방법은 다른 방법에서는 효과가 있었지만, 프리트레인에서는 거의 효과가 없었음.
그래서 프리트레인의 대표주자인 BERT를 써서 성능 향상되는지 확인.

## 1.3 지금까지 결과 분석(2022.06.01)
CNN과 GRU에서는 확실히 효과가 보이지만 BERT에서는 2,000개만 되도 성능이 열화되는 듯...
다행인건 CNN에서 크게는 정확도에서 7% 이상의 성능 향상이 보이기도 함. 