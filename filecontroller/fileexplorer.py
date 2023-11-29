import os
from typing import List


# .py 랑 .java확장자인 파일을 모으는 역할을 하는 클래스
class Explorer:

    # root_dir: 파일을 탐색할 디렉토리이고 str형이다.

    def __init__(self, root_dir: str):
        self.paths = []
        self.root = root_dir
        self.explorer_file(root_dir)
        # self.check_C = False

    def explorer_file(self, root):
        # listdir(path) 특정 경로 내에 존재하는 폴더(디렉토리)와 파일 리스트를 검색
        files = os.listdir(root)
        for file in files:
            path = os.path.join(root, file)
            if os.path.isdir(path):
                self.explorer_file(path)
            else:
                # if os.path.splitext(path)[1] == '.c' or os.path.splitext(path)[1] == '.h' or os.path.splitext(path)[1] == '.java' or os.path.splitext(path)[1] == '.py':
                #     self.paths.append(path)89089
                if os.path.splitext(path)[1] == '.py' and '__init__' not in path:  # splitext: 파일명과 확장자를 분리해줌  text.txt -> text, .txt
                    self.paths.append(path)
                elif os.path.splitext(path)[1] == '.java':
                    self.paths.append(path)
                elif os.path.splitext(path)[1] == '.c':
                    # self.check_C = True
                    self.paths.append(path)
                else:
                    continue

    def get_paths(self) -> List:
        return self.paths
