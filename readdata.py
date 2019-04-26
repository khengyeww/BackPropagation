
"""
#f = open("iris.data", "r")
#f = open("bezdekIris.data", "r")
print(f.read())
f.close()
"""

def remove_blank_lines(file):
    for l in file:
        line = l.rstrip()
        if line:
            yield line

# データの読み込み
def read_data(filename = ""):
	if filename == "":
		filename = "data/iris.data"

	array = []

	# データを行ごとに配列に追加
	with open(filename) as file:
		for line in remove_blank_lines(file):
			array.append(line.rstrip().split(','))
			
	# 配列をそのまま出力
	# # print(array)
	
	# 配列を行ごとに出力
	# for row in array:
	# 	print(row[-1])
	
	# for row in array:
	# 	row.insert(len(row), row.pop(0)) 
	
	return array