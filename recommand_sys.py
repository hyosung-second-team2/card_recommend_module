from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def cosine_measure(v1, v2):
    prod = np.dot(v1, v2)
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    return prod / (len1 * len2)

def jacaard_similarity(listA, listB):
    s1 = set(listA)
    s2 = set(listB)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


######################################################################################################################
#        유사도 알고리즘은 카드 추천 시스템에 적합하지 않아 폐기                                                              #
#        예) [주유, 교통, 외식, 연회비지원] 4개의 카테고리를 가진 카드A 와, [주유, 교통] 혜택을 가진 카드 B                       #
#        이때, 사용자 C가 주유, 교통에 지출이 많을 때, 카드 A가 혜택이 많다는 이유로 유사도가 카드 B보다 떨어지는 문제가 있다.        #
#따라서, 각 카드사의 카드의 카테고리 별 혜택(통합할인한도, 할인률 등)을 직접 수집하여 계산하는 것으로 변경하였다. recommend_calc.py참고#
######################################################################################################################

user_tendency = ['대형마트, 교통, 카페/베이커리']
cards_rewards = [
    ['주유 교통 외식 연회비지원'],
    ['연회비지원 외식 편의점'],
    ['문화 레저 항공마일리지'],
    ['대형마트 문화 레저 교통 편의점 연회비지원 항공마일리지 외식 유치원 육아 바나나 사과 오이 마라탕 핸드폰 스마트폰 편의점']
]

print(jacaard_similarity(user_tendency, cards_rewards[0]))
print(jacaard_similarity(user_tendency, cards_rewards[1]))
print(jacaard_similarity(user_tendency, cards_rewards[2]))
print(jacaard_similarity(user_tendency, cards_rewards[3]))

count_vec = CountVectorizer()
doc_term_matrix = count_vec.fit_transform([user_tendency[0], cards_rewards[0][0], cards_rewards[1][0], cards_rewards[2][0], cards_rewards[3][0]]).toarray()
print(doc_term_matrix)

print(cosine_measure(doc_term_matrix[0], doc_term_matrix[1]) * 100)
# 0.7071067811865476
print(cosine_measure(doc_term_matrix[0], doc_term_matrix[2]) * 100)
# 0.0
print(cosine_measure(doc_term_matrix[0], doc_term_matrix[3]) * 100)
# 0.0
print(cosine_measure(doc_term_matrix[0], doc_term_matrix[4]) * 100)


