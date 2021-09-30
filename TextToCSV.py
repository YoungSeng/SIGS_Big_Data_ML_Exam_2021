
# import pandas as pd
import csv

root = r'C:\HCSI\SiCheng Yang\杨思程\我的资料库\我的坚果云\课程\大数据机器学习\第一次实验\\'
txt_path = root + 'result.txt'
csv_path = root + 'submission.csv'

with open(csv_path, 'w+', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['Id', 'Expected'])

    res = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        for line in file:
            res.append([x for x in line.strip().split(',')])
            # res.append([line])

    # print(res)
    writer.writerows(res)



