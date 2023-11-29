from exception import JFileReadError


# 파일 경로를 입력으로 받아 해당 파일을 읽어들이고, 읽어들인 내용을 self.source 변수에 저장합니다.
# 또한 파일 경로에서 파일 이름과 확장자를 추출하여 _ 문자를 이용해 새로운 파일 경로를 생성하고, 생성된 파일 경로를 self.write_path 변수에 저장
class Reader:

    def __init__(self, file_path: str):
        self.source = ''
        self.write_path = ''
        paths = file_path.split('/')
        if len(paths) < 2:
            raise JFileReadError("File Read Error")

        for path in paths:
            if path == '.' or path == '..':
                continue
            if '.' in path and len(path) > 1:
                extend = path.split('.')
                self.write_path = self.write_path + '_' + extend[0] + '.txt'
            else:
                if self.write_path == '':
                    self.write_path = path
                else:
                    self.write_path = self.write_path + '_' + path

        source = open(file_path, 'r', encoding='ISO-8859-1')
        self.source = source.read()
        source.close()

    # self.source에 저장된 내용을 반환
    def get_source(self) -> str:
        return self.source

    # self.write_path에 저장된 파일 경로를 반환
    def get_write_path(self) -> str:
        return self.write_path
