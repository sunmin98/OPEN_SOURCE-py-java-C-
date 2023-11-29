from gensim.models import Word2Vec
from lexer import Embedder

class W2V:
    def __init__(self):
        self.model = None
        self.tokens = Embedder().embedding_dict
        # tokens = [key for key, value in self.tokens.items()]
        # print(tokens)
        # print(len(tokens))


#단어 임베딩을 학습하는 함수.
    def train(self, embedding_result):
        print(embedding_result)
        self.model = Word2Vec(
            sentences=embedding_result,
            vector_size=100,#벡터 크기
            window=5, #컨텍스트 윈도우 크기
            hs=1, # 계층적 소프트맥스 사용여부 --> 스킵 그램
            min_count=1, #단어 최소 빈도수 제한(빈도수가 적은 단어들은 학습하지 않는다.)
            workers=2 #학습을 위한 프로세스 수
        )

        self.model.save('word2vec.model')
        print("말뭉치 개수 ->", self.model.corpus_count)
        print("말뭉치 내 전체 단어수 ->", self.model.corpus_total_words)

#소스 벡터화하는 함수 --> 토큰을 벡터화
    def get_source_vectors(self, source, model):
        source_embedding_list = []
        doc2vec = None
        count = 0
        for line in source.split('\n'):
            if line in model.wv.key_to_index:
                count += 1
                if doc2vec is None:
                    doc2vec = model.wv[line]
                else:
                    doc2vec = doc2vec + model.wv[line]

        if doc2vec is not None:
            doc2vec = doc2vec / count
            source_embedding_list.append(doc2vec)

        return source_embedding_list, doc2vec