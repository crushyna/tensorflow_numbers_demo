from time import sleep
from tkinter import *
from tkinter import ttk, messagebox
import PIL
from PIL import ImageGrab, Image
from number_recognizer_2 import NeuralNetwork

filename = 'None!'


class Main:
    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 40
        self.draw_widgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penwidth, fill=self.color_fg,
                               capstyle=ROUND, smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None

    def changewidth(self, e):
        self.penwidth = e

    def save(self):
        sleep(1)
        # file specs
        save_width = 28
        save_height = 28
        filetitle = 'image.png'
        global filename
        filename = filetitle

        # save file
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        PIL.ImageGrab.grab().crop((x, y, x1, y1)).save(filetitle)

        # resize image
        image_in = PIL.Image.open(filetitle)
        image_out = image_in.resize((save_width, save_height), PIL.Image.ANTIALIAS)
        image_out.save(filetitle)
        return filename

    def clear(self):
        self.c.delete(ALL)

    def recognize_number(self):
        self.save()
        recognized_number = NeuralNetwork.predict_number(filename)
        result_answer1 = messagebox.askyesno("Number recognition", f"Recognized number: {recognized_number}. \n"
                                                                   f"Is this number correct?")
        if result_answer1:
            return recognized_number
        else:
            # TODO: this needs some kind of GUI implemented
            proper_number = int(input("Enter proper number: "))
            NeuralNetwork.merge_images(proper_number, 'image.png')

    def train_model(self):
        messagebox.showinfo("Training", "This might take few minutes, please wait.")
        NeuralNetwork.train_neural_network()
        messagebox.showinfo("Training", "Training completed!")

    def close_window_and_retrain(self):
        self.win.quit()

    def draw_widgets(self):
        self.controls = Frame(self.master, padx=5, pady=5)
        Label(self.controls, text='Pen Width: ', font=('', 10)).grid(row=0, column=0)
        self.slider = ttk.Scale(self.controls, from_=15, to=50, command=self.changewidth, orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0, column=1, ipadx=30)
        self.controls.pack()
        self.c = Canvas(self.master, width=280, height=280, bg=self.color_bg, )
        self.c.pack(fill=BOTH, expand=True)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Save', command=self.save)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear canvas', command=self.clear)
        optionmenu.add_command(label='Recognize digit', command=self.recognize_number)
        optionmenu.add_command(label='Train neural network', command=self.train_model)
        optionmenu.add_command(label='Exit', command=self.master.destroy)


if __name__ == '__main__':
    root = Tk()
    Main(root)
    root.title('DrawingApp')
    root.mainloop()
