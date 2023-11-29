from lexer import Lexer, Embedder
from filecontroller import Reader, Writer, Explorer
from measure_copy_value import Measure, CheckSourceRoutine
from precheck import PreCheck, W2V, D2V
import sys
from sentence_transformers import SentenceTransformer, util
import time
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import pandas as pd


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# 주어진 소스코드를 읽어들이고, Lexer 클래스를 사용하여 토큰화된 결과를 얻어 임베딩해서 임베딩한 토큰을 파일로 저장.
def create_opensource_token_process(source: str, save: str, token_save_path: str):
    reader = Reader(source)
    # 파일 경로를 입력으로 받아 해당 파일을 읽어들이고, 읽어들인 내용을 self.source 변수에 저장합니다.
    # 또한 파일 경로에서 파일 이름과 확장자를 추출하여 _ 문자를 이용해 새로운 파일 경로를 생성하고, 생성된 파일 경로를 self.write_path 변수에 저장
    # print("reader -- > ",reader.get_source())
    lexer = Lexer(reader.get_source())
    # print("lexer -- > ",lexer)
    tokens = lexer.lex()
    # print("tokens -- > ",tokens)
    token = '\n'.join(map(str, tokens))

    embedder = Embedder()
    writer = Writer(reader.get_write_path(), save, embedder.get_embedding(tokens), token_save_path, token)
    # Writer 클래스는 토큰을 파일로 저장하는 클래스
    return writer.get_token_list()
    # return TaggedDocument(tags=[reader.get_write_path()], words=writer.get_token_list()) --> ???


def create_opensource_token_file():
    if len(sys.argv) != 4:  # 명령어 단어가 4개이여야함
        print('python main.py/ open_source_path/ save_embedding_path/ save_token_path')
        sys.exit()
    print(sys.argv)
    # model = SentenceTransformer('all-MiniLM-L6-v2')  -> 문장이 유사도를 보여주는 모델
    explorer = Explorer(sys.argv[1])
    explorer_paths = explorer.get_paths()  # 주소입력받은걸 변수 저장
    sentence = []
    print("explorer_paths", explorer_paths)
    # print("explorer_paths==", explorer_paths[1])
    for path in enumerate(explorer_paths):
        print("패뜨", path)
        # print("check_c--", check_c)
    for path in explorer_paths:
        print(path)
        sentence.append(
            create_opensource_token_process(path, save=sys.argv[2], token_save_path=sys.argv[3]))
    print("sentence")
    print(sentence)
    w2v(sentence)
    # documents = []
    #
    # for path in explorer_paths:
    #     print(path)
    #     documents.append(create_opensource_token_process(path, save=sys.argv[2], token_save_path=sys.argv[3]))
    # d2v(documents)

def create_opensource_token_file1():
    # if len(sys.argv) != 4:  # 명령어 단어가 4개이여야함
    #     print('python main.py/ open_source_path/ save_embedding_path/ save_token_path')
    #     sys.exit()
    lists=['test.py' ,'netcat-cpi-kernel-module-master'  ,'test_save_emvedding' , 'test_save_token']
    print(lists)
    # model = SentenceTransformer('all-MiniLM-L6-v2')  -> 문장이 유사도를 보여주는 모델
    explorer = Explorer(lists[1])
    explorer_paths = explorer.get_paths()  # 주소입력받은걸 변수 저장
    sentence = []
    print("explorer_paths", explorer_paths)
    # print("explorer_paths==", explorer_paths[1])
    for path in enumerate(explorer_paths):
        print("패뜨", path)
        # print("check_c--", check_c)
    for path in explorer_paths:
        print(path)
        sentence.append(
            create_opensource_token_process(path, save=sys.argv[2], token_save_path=sys.argv[3]))
    print("sentence")
    print(sentence)
    w2v(sentence)
    # documents = []
    #
    # for path in explorer_paths:
    #     print(path)
    #     documents.append(create_opensource_token_process(path, save=sys.argv[2], token_save_path=sys.argv[3]))
    # d2v(documents)

# 토큰 만드는거
def create_source_token(path: str):
    reader = Reader(path)
    lexer = Lexer(reader.get_source())
    tokens = lexer.lex()
    print("입력 받은 소스를 토큰으로 만들었다~! ", tokens)
    return '\n'.join(map(str, tokens))


def measure_process():
    if len(sys.argv) != 2:
        print('python main.py source_path')

    explorer = Explorer(sys.argv[1])
    explorer_paths = explorer.get_paths()

    for path in explorer_paths:
        print(path)
        start = time.time()
        measure = Measure(create_source_token(path))
        measure.measure('Python')  # 언어 설정!!!
        print(f'source_file: {path} / score: {measure.pair[0]} / taken time: {time.time() - start:.5f} sec')
        # print(f'source_file: {path} / score: {measure.pair[len(measure.pair)-1]}')
        # print(measure.pair)


def test():
    reader = Reader('./OpenSource/blockchain-python/account.py')
    reader2 = Reader('./test_source/gameRole.py')
    # model = SentenceTransformer('krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net')
    model = SentenceTransformer('krlvi/sentence-t5-base-nlpl-code_search_net')
    # model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')
    # model = SentenceTransformer('mchochlov/codebert-base-cd-ft')
    # reader1_embedding = model.encode(reader.get_source(), convert_to_tensor=True)
    # reader2_embedding = model.encode(reader2.get_source(), convert_to_tensor=True)
    reader1_embedding = model.encode(reader.get_source())
    reader2_embedding = model.encode(reader2.get_source())
    value = util.cos_sim(reader1_embedding, reader2_embedding)
    print(value, value[0], value[0][0])
    if value[0][0] > 0.75:
        print('over')
    else:
        print('lower')


def pre_check():
    if len(sys.argv) != 2:
        print('python main.py source_path')

    explorer = Explorer(sys.argv[1])
    explorer_paths = explorer.get_paths()
    result = ''

    for path in explorer_paths:
        print(path)
        precheck = PreCheck(create_source_token(path))
        precheck.precheck('Python')
        result = result + '\n' + f'source_file: {path} / most_score: {precheck.get_most_value_pair()}'
        # print(f'source_file: {path} / most_score: {precheck.get_most_value_pair()}')
    save_result(result)


def save_result(result: str):
    file = open('./result/result.txt', 'w')
    file.write(result)
    file.close()


def w2v(sentence):
    wv = W2V()
    wv.train(sentence)  # 단어 임베딩을 학습하는 함수.


def d2v(documents):
    dc = D2V(documents)


def w2v_test():
    model = Word2Vec.load('word2vec.model')
    result = f'IMPORT : {model.wv["IMPORT"]}'
    print(result)
    file = open('embedding_value_example.txt', 'w')
    file.write(result)

    print(model.wv.index_to_key)


def w2v_pretest():
    model = Word2Vec.load('word2vec.model')
    source1 = create_source_token('validate_code/Python/Copy/account.py')
    source2 = create_source_token('validate_code/Python/No_relation/parser.py')
    wv = W2V()
    _, doc2vec1 = wv.get_source_vectors(source1, model)
    _, doc2vec2 = wv.get_source_vectors(source2, model)
    print(cos_sim(doc2vec1, doc2vec2))


# 실제 실행 코드.
def main():
    print("메인 함수입니당")
    if len(sys.argv) != 2:
        print('python main.py source_path(생파일)')   #생 파일 입력.
        exit()
        #    xxx xxxx 이렇게 해서 실행하는것.
        # 따라서 2개의 인수가 아니면 오류 표시하고 종료.

    print(sys.argv)
    main_start_time = time.time()
    explorer = Explorer(sys.argv[1])
    print(sys.argv[1])
    print("explorer-->", explorer)
    print("읽을 파일 주소->", explorer.get_paths())
    explorer_paths = explorer.get_paths()

    print(f'file count : {len(explorer_paths)}')
    result = []

    for path in explorer_paths:
        # path = 내가입력한 생코드
        print('path-->', path)
        start = time.time()
        precheck = PreCheck(create_source_token(path)) #생 코드를 토큰화함.
        print("precheck --> ", precheck)
        precheck.precheck("C-TOKEN")   #토큰 만 있는 폴더   #토큰을 임볘딩하고 유사도를 계산함
        print("~~~~~~~~~~~~~~~~~~~~")
        # precheck.precheck('Java')  ## Guesslang 이식을 하다가 말아서. 소스가 무슨 언어로 되어있는지 판별이 추가가 되어야 함.
        # ptrcheck에는 토큰만
        print("precheck확인", precheck.result)
        print("Reader(path).get_source() 111" , Reader(path).get_source())
        check = CheckSourceRoutine(Reader(path).get_source(), precheck.result)
        # 원본 소스 코드
        # ./OpenSource/{language

        check.check_value("OepnSource_C", "C")  #  토큰화해놓은 원본 코드 파일
        #Python , JAVA , C
        # 2. path를 받지않는다,사용하지않는다

        # check.check_value("유사도 측정당할 토큰모아놓는 파일 이름주소 입력", path)
        # check.check_value('./OpenSource', 'Java')  ## forder 이름 맞춰주기.

        # print("check_value ===> ",check.check_value("test_save_emvedding", "test_save_emvedding"))

        print(f'source_file: {path} / most_score: {check.value}')
        print("check.result-->", check.result)
        if len(check.result) > 0:
            print("검출~~검출")
            # 1. 이건 검출될때의 코드가 아닌가.?
            print(f'source_file: {path} / most_score: {check.result[0]} / taken time : {time.time() - start:.5f} sec')
            result.append({'source_file': path, 'value': check.result, 'taken time': f'{time.time() - start:.5f} sec'})
        else:
            print("미검출")
            print(f'source_file: {path} / [], taken time : {time.time() - start:.5f} sec')
            result.append({'source_file': path, 'value': '[]', 'taken time': f'{time.time() - start:.5f} sec'})
        print("=============================================================================================================")
        # result.append({'source_file': path, 'value': check.result[0], 'taken time': f'{time.time() - start:.5f} sec'})
    df = pd.DataFrame(result)
    # print(df)
    csv_name = input("csv 파일 이름을 입력하세요 --> ")
    df.to_csv(csv_name)
    print(f'total : {time.time() - main_start_time:.5f} sec')

if __name__ == '__main__':
    print("start")
    # create_opensource_token_file()   #오픈소스만을 토큰화하는겨

    ### sys.argv= ['main.py', 'netcat-cpi-kernel-module-master',"test_save_emvedding", "test_save_token"]
    ### create_opensource_token_file(sys.argv[1], sys.argv[2], sys.argv[3])
    ### netcat-cpi-kernel-module-master test_save_emvedding test_save_token

    main()
    ### python main.py test_source   ---> 오픈소스가 섞여있는 생코드를 입력 , 생코드에 오픈소스가 얼마나 섞여있는지 확인하기위해.
