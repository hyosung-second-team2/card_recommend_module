유사도 알고리즘은 카드 추천 시스템에 적합하지 않아 폐기

적합하지 않은 예) 
[주유, 교통, 외식, 연회비지원] 4개의 카테고리를 가진 카드A 와, [주유, 교통] 혜택을 가진 카드 B 가 있다고 가정
이때, 사용자 C가 주유, 교통에 지출이 많을 때, 카드 A가 혜택이 많다는 이유로 유사도가 카드 B보다 떨어지는 문제가 있다.        
따라서, 각 카드사의 카드의 카테고리 별 혜택(통합할인한도, 할인률 등)을 직접 수집하여 사용자의 지출내역을 기반으로 모든 카드의 피킹률 및 할인액을 계산하는 것으로 변경하였다. 
recommend_calc.py참고#

[주의사항] 코드 중 oracledb.init_oracle_client(lib_dir="C:/hyosungedu/DevUtils/instantclient_21_13") 
이 부분은 오라클 instantclient를 다운 받고 노트북의 다운받은 경로를 작성해줘야함. os에 맞게 다운 받고 나서 수정!
https://www.oracle.com/database/technologies/instant-client/downloads.html
