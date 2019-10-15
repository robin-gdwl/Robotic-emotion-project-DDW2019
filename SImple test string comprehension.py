text = "hallo leute"

for fragment in text:
    for letter in fragment:
        print("letter:", letter)
        #motion_path.extend(self.letter_to_coordinates(letter, [x_offset, y_offset]).copy())
        #x_offset += offset_amount
        #print("x_offset", x_offset)
        letter = letter*2
        print(letter)

print(text)