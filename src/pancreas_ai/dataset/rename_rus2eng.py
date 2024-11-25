from collections import OrderedDict
import os

DS_PATH = "c:\\Users\\ikuchin\\Downloads\\pancreas_data\\ds1-pomc\\"

ru2eng = OrderedDict([
    ("а", "a"),
    ("б", "b"),
    ("в", "v"),
    ("г", "g"),
    ("д", "d"),
    ("е", "e"),
    ("ё", "yo"),
    ("ж", "zh"),
    ("з", "z"),
    ("и", "i"),
    ("й", "y"),
    ("к", "k"),
    ("л", "l"),
    ("м", "m"),
    ("н", "n"),
    ("о", "o"),
    ("п", "p"),
    ("р", "r"),
    ("с", "s"),
    ("т", "t"),
    ("у", "u"),
    ("ф", "f"),
    ("х", "kh"),
    ("ц", "ts"),
    ("ч", "ch"),
    ("ш", "sh"),
    ("щ", "sch"),
    ("ъ", ""),
    ("ы", "y"),
    ("ь", ""),
    ("э", "e"),
    ("ю", "yu"),
    ("я", "ya"),
    (" ", "_"),
    ("(", "_"),
    (")", "_"),
    (",", "_"),
    ("-", "_"),
])

def to_lower_case(s):
    return s.lower()

def __ru2eng(s):
    for ru, eng in ru2eng.items():
        s = s.replace(ru, eng)
    return s

def rename_folders_recursively(path):

    rename_flag = True
    iter = 0

    while rename_flag:
        rename_flag = False
        iter += 1
        print(f"Iteration: {iter}")

        for root, dirs, files in os.walk(path):
            for dir in dirs:
                new_dir = __ru2eng(to_lower_case(dir))

                if dir != new_dir:
                    os.rename(os.path.join(root, dir), os.path.join(root, new_dir))
                    print(f"Renamed: {os.path.join(root, dir)} -> {new_dir}")

                    rename_flag = True

    print(f"Total iterations: {iter-1}")
    return

def rename_files_recursively(path):

    rename_flag = True
    iter = 0

    while rename_flag:
        rename_flag = False
        iter += 1
        print(f"Iteration: {iter}")

        for root, dirs, files in os.walk(path):
            for file in files:
                new_file = __ru2eng(to_lower_case(file))

                if file != new_file:
                    os.rename(os.path.join(root, file), os.path.join(root, new_file))
                    print(f"Renamed: {os.path.join(root, file)} -> {new_file}")

                    rename_flag = True

    print(f"Total iterations: {iter-1}")
    return

def main():
    rename_folders_recursively(DS_PATH)
    rename_files_recursively(DS_PATH)

    return

if __name__ == "__main__":
    main()
