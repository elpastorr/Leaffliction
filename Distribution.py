#!/usr/bin/env python3
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("TKAgg")


def get_category(images_dir):
    category = dict()
    for dir_name in os.listdir(images_dir):
        try:
            if not os.path.isdir(os.path.join(images_dir, dir_name)):
                raise Exception("Input is not a directory of directories")
        except Exception as e:
            print(e.__class__.__name__, e)
            exit(0)
        nb_files = len(os.listdir(os.path.join(images_dir, dir_name)))
        if dir_name.startswith('Apple_'):
            dir_name = 'A_' + dir_name[6:]
        elif dir_name.startswith('Grape_'):
            dir_name = 'G_' + dir_name[6:]
        category[dir_name] = nb_files
    return category


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise Exception("Input should be: Distribution.py images_dir")
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(0)
    try:
        if not os.path.isdir(sys.argv[1]):
            raise Exception("Input is not a directory")
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(0)

    category = get_category(sys.argv[1])
    dir_names = list(category.keys())
    images_nb = list(category.values())
    fruit = sys.argv[1].split('/')[-1]

    figure, axes = plt.subplots(ncols=2)
    color = sns.color_palette('tab10', len(dir_names))
    figure.suptitle(fruit + ' class distribution')

    axes[0].pie(images_nb, labels=dir_names, autopct='%.1f%%')
    axes[1].bar(category.keys(), category.values(), color=color)
    plt.show()
