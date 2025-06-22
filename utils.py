from pathlib import Path
import shutil

def move_files(file_list: list, dst_dir: str, suffix: str = "") -> None:
    """
    파일 리스트에 있는 모든 파일을 대상 디렉토리로 이동하고 선택적으로 파일명에 접두사/접미사를 추가합니다.
    
    Args:
        file_list (list): 이동할 파일 경로들의 리스트 (str 또는 Path 객체)
        dst_dir (str): 파일들을 이동할 대상 디렉토리
        prefix (str, optional): 파일명 앞에 추가할 문자열
        suffix (str, optional): 파일 확장자 앞에 추가할 문자열
    """
    if isinstance(file_list, (str, Path)):
        file_list = [file_list]

    dst = Path(dst_dir)

    # 대상 디렉토리 확인 및 생성
    if not dst.exists():
        try:
            dst.mkdir(parents=True, exist_ok=True)
            print(f"정보: 대상 디렉토리를 생성합니다: {dst.absolute()}")
        except Exception as e:
            print(f"디렉토리 생성 중 오류 발생: {e}")
            return
    elif not dst.is_dir():
        print(f"오류: 대상 경로가 디렉토리가 아닙니다: {dst.absolute()}")
        return

    # 이동 성공/실패 카운트
    success_count = 0
    failed_count = 0
    
    for file_path in file_list:
        if isinstance(file_path, str):
            file_path_s = file_path
            file_path_p = Path(file_path)
        elif isinstance(file_path, Path):
            file_path_s = str(file_path)
            file_path_p = file_path
        else:
            raise ValueError(f"file_path 의 타입이 올바르지 않습니다: {type(file_path)}")
        
        try:
            # 새로운 파일명 생성
            original_stem = file_path_p.stem  # 확장자를 제외한 파일명
            original_suffix = file_path_p.suffix  # 확장자
            new_filename = f"{original_stem}_{suffix}.{original_suffix}"
            destination_path = dst / new_filename

            # 파일 이동
            shutil.move(file_path_s, str(destination_path))
            success_count += 1
        except Exception as e:
            failed_count += 1

    print(f"\n이동 완료 요약:")
    print(f"성공: {success_count}개 파일")
    print(f"실패: {failed_count}개 파일")
    print(f"대상 디렉토리: {dst.absolute()}")
    
    return


def move_directory(source_dir: str, destination_parent_dir: str) -> None:
    source = Path(source_dir)
    # The destination directory should be the parent where the source will be moved *into*
    destination_parent = Path(destination_parent_dir)

    if not source.is_dir():
        print(f"오류: 소스 경로가 디렉토리가 아니거나 존재하지 않습니다: {source.absolute()}")
        return

    if not destination_parent.exists():
        print(f"정보: 대상 상위 디렉토리가 존재하지 않아 생성합니다: {destination_parent.absolute()}")
        try:
            destination_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"대상 상위 디렉토리 생성 중 오류 발생: {e}")
            return
    elif not destination_parent.is_dir():
        print(f"오류: 대상 경로가 디렉토리가 아닙니다: {destination_parent.absolute()}")
        return

    # Check if a directory with the same name already exists in the destination
    destination_path = destination_parent / source.name
    if destination_path.exists():
         print(f"오류: 대상 디렉토리에 이미 같은 이름의 파일 또는 디렉토리가 존재합니다: {destination_path.absolute()}")
         return

    try:
        shutil.move(str(source), str(destination_parent))
        print(f"디렉토리 이동 완료: {source.absolute()} -> {destination_parent.absolute()   }")
    except Exception as e:
        print(f"디렉토리 이동 중 오류 발생: {e}")

    return

def copy_files(file_list: list, dst_dir: str, suffix: str = "") -> None:
    """
    파일 리스트에 있는 모든 파일을 대상 디렉토리로 복사하고 선택적으로 파일명에 접두사/접미사를 추가합니다.
    
    Args:
        file_list (list): 복사할 파일 경로들의 리스트
        dst_dir (str): 파일들을 복사할 대상 디렉토리
        prefix (str, optional): 파일명 앞에 추가할 문자열
        suffix (str, optional): 파일 확장자 앞에 추가할 문자열
    """
    if isinstance(file_list, (str, Path)):
        file_list = [file_list]

    dst = Path(dst_dir)

    # 대상 디렉토리 확인 및 생성
    if not dst.exists():
        try:
            dst.mkdir(parents=True, exist_ok=True)
            print(f"정보: 대상 디렉토리를 생성합니다: {dst.absolute()}")
        except Exception as e:
            print(f"디렉토리 생성 중 오류 발생: {e}")
            return
    elif not dst.is_dir():
        print(f"오류: 대상 경로가 디렉토리가 아닙니다: {dst.absolute()}")
        return

    # 복사 성공/실패 카운트
    success_count = 0
    failed_count = 0
    
    for file_path in file_list:
        if isinstance(file_path, str):
            file_path_s = file_path
            file_path_p = Path(file_path)
        elif isinstance(file_path, Path):
            file_path_s = str(file_path)
            file_path_p = file_path
        else:
            raise ValueError(f"file_path 의 타입이 올바르지 않습니다: {type(file_path)}")
        
        try:
            # 새로운 파일명 생성
            original_stem = file_path_p.stem  # 확장자를 제외한 파일명
            original_suffix = file_path_p.suffix  # 확장자
            new_filename = f"{original_stem}_{suffix}.{original_suffix}"
            destination_path = dst / new_filename

            # 파일 복사
            shutil.copy2(file_path_s, str(destination_path))
            success_count += 1
            print(f"파일 복사 완료: {file_path_p.name} -> {new_filename}")
        except Exception as e:
            print(f"파일 복사 중 오류 발생 ({file_path_p.name}): {e}")
            failed_count += 1

    print(f"\n복사 완료 요약:")
    print(f"성공: {success_count}개 파일")
    print(f"실패: {failed_count}개 파일")
    print(f"대상 디렉토리: {dst.absolute()}")
    
    return

def copy_directory(source_dir: str, destination_parent_dir: str) -> None:
    source = Path(source_dir)
    destination_parent = Path(destination_parent_dir)

    if not source.is_dir():
        print(f"오류: 소스 경로가 디렉토리가 아니거나 존재하지 않습니다: {source.absolute()}")
        return

    if not destination_parent.exists():
        print(f"정보: 대상 상위 디렉토리가 존재하지 않아 생성합니다: {destination_parent.absolute()}")
        try:
            destination_parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"대상 상위 디렉토리 생성 중 오류 발생: {e}")
            return
    elif not destination_parent.is_dir():
        print(f"오류: 대상 경로가 디렉토리가 아닙니다: {destination_parent.absolute()}")
        return

    # 대상 경로에 같은 이름의 디렉토리가 있는지 확인
    destination_path = destination_parent / source.name
    if destination_path.exists():
        print(f"오류: 대상 디렉토리에 이미 같은 이름의 파일 또는 디렉토리가 존재합니다: {destination_path.absolute()}")
        return

    try:
        shutil.copytree(str(source), str(destination_path))  # copytree는 디렉토리와 그 내용을 재귀적으로 복사
        print(f"디렉토리 복사 완료: {source.absolute()} -> {destination_path.absolute()}")
    except Exception as e:
        print(f"디렉토리 복사 중 오류 발생: {e}")

    return

