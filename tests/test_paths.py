from paths import normalize_path_basename


def test_normalize_path_basename_windows_absolute_path():
    path = r"C:\Users\Nafeu\dronmakr-files\exports\folysplitr___exports___split___0.wav"
    assert normalize_path_basename(path) == "folysplitr___exports___split___0.wav"


def test_normalize_path_basename_posix_path():
    path = "/home/user/dronmakr-files/exports/sample.wav"
    assert normalize_path_basename(path) == "sample.wav"


def test_normalize_path_basename_mixed_export_token():
    path = r"/exports/C:\Users\Nafeu\dronmakr-files\exports\sample.wav"
    assert normalize_path_basename(path) == "sample.wav"
