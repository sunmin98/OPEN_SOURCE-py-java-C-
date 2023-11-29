from sentence_transformers import SentenceTransformer, util
import os
from exception import JMeasureError
from filecontroller import Reader, Explorer
from typing import List
from .config import THRESHOLD
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel


# 소스간의 유사도를 측정하는것
class CheckSourceRoutine:

    def __init__(self, source: str, word_embedding_value: List[dict]):
        self.source = source
        self.code_bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.model = SentenceTransformer('krlvi/sentence-t5-base-nlpl-code_search_net')
        self.java_model = SentenceTransformer('ncoop57/codeformer-java')
        # self.C_model = SentenceTransformer('ncoop57/codeformer-c')
        self.value = word_embedding_value
        self.result = []

    # 주어진 root 경로에 있는 모든 파일을 탐색하면서, 파일 이름과 비교 대상 소스코드,
    # 소스코드와의 유사도 값을 기반으로 검증할 소스코드와 유사한 소스코드 파일을 찾아서 그 결과를 리스트 형태로 반환합니다
    def check_value(self, root: str, lang: str):
        explorer = Explorer(root)
        print("explorer 학인 -> ", explorer)
        explorer_paths = explorer.get_paths()
        print("explorer_paths 학인 -> ", explorer_paths)

        for path in explorer_paths:
            print(f'path 패뜨: {path}')
            tmp: list = path.split('.')
            key: str = tmp[len(tmp) - 2][:].replace('/', '_')
            target = Reader(path)
            print("self.value 확인 ->", self.value)
            for value in self.value:
                print("value['file'] 확인", value['file'])
                print("key 확인", key)
                print("")

                if value['file'] == key:  # 파일 이름이 같은 경우
                    # print(f'value: {value["score"]}')
                    # print(f'cos: {util.cos_sim(self.model.encode(self.source), self.model.encode(target.get_source()))}')
                    # self.value['score'] = value['score'] * util.cos_sim(self.model.encode(self.source), self.model.encode(target.get_source()))[0][0]
                    # if lang == 'Python':
                    #     cos_value = value['score'] * util.cos_sim(self.model.encode(self.source), self.model.encode(target.get_source()))[0][0]
                    # elif lang == 'Java':
                    #     cos_value = value['score'] * util.cos_sim(self.java_model.encode(self.source), self.java_model.encode(target.get_source()))[0][0]
                    # else:
                    #     cos_value = 0.0

                    # if lang == "Python":
                    #     cos_value = value['score'] * \
                    #                 util.cos_sim(self.model.encode(self.source),
                    #                              self.model.encode(target.get_source()))[0][0]
                    #     print("파이썬  cos_value 확인 ~~~~~~~~~~~~~~>", cos_value)
                    #
                    # if lang == "JAVA":
                    #     cos_value = value['score'] * \
                    #                 util.cos_sim(self.java_model.encode(self.source),
                    #                              self.java_model.encode(target.get_source()))[0][0]
                    #     print("자바  cos_value 확인 ~~~~~~~~~~~~~~>", cos_value)
                    #
                    # if lang == "C":
                    #     cos_value = value['score'] * \
                    #                 util.cos_sim(self.C_model.encode(self.source),
                    #                              self.C_model.encode(target.get_source()))[0][0]
                    #     print("씨  cos_value 확인 ~~~~~~~~~~~~~~>", cos_value)
                    # if lang == "CODE_BERT":
                    #     cos_value = value['score'] * \
                    #                 util.cos_sim(self.code_bert_model.encode(self.source),
                    #                              self.code_bert_model.encode(target.get_source()))[0][0]
                    #     print("코드버트  cos_value 확인 ~~~~~~~~~~~~~~>", cos_value)
                    cos_value = value['score'] * \
                                util.cos_sim(self.model.encode(self.source), self.model.encode(target.get_source()))[0][
                                    0]
                    if cos_value >= 0.5:
                        self.result.append(
                            {
                                'file': value['file'],
                                'score': cos_value
                            }
                        )
                        break
        self.result = sorted(self.result, key=lambda x: x['score'], reverse=True)
