def print_unicode_characters_in_range(a, b):
	ch=a
	while(ch<=b):
		print(ch, end="  ")
		ch = chr(ord(ch)+1)

	#also print special characters.
	
	l=["%" , "/", "?", ":", ",",".","-"]
	for a in l:
		print(a , end="  ")


if __name__ == "__main__":
	print_unicode_characters_in_range('ऀ',"ॲ")
	
