from os import listdir
from time import sleep
from tkinter import *
from tkinter import ttk, messagebox, simpledialog
import PIL
from PIL import ImageGrab, Image
from number_recognizer_3 import NeuralNetwork

filename = 'None!'


class Main:
    def __init__(self, master):
        self.counter = 0
        self.master = master
        self.color_fg = 'black'
        self.color_bg = '#ffffff'
        self.old_x = None
        self.old_y = None
        self.penwidth = 32.5
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
        filetitle = 'image.bmp'
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

        # save working dataset
        network1.save_working_dataset()

        return filename

    def clear(self):
        self.c.delete(ALL)

    def recognize_number(self):
        self.save()
        recognized_number = network1.predict_number(filename)
        result_answer1 = messagebox.askyesno("Number recognition", f"Recognized number: {recognized_number}. \n"
                                                                   f"Is this number correct?")
        if result_answer1:
            return recognized_number
        else:
            inp = InputBox(text="Enter proper number and press enter: ")
            print(inp.get)
            network1.merge_images(inp.get, filename)
            print('Image appended to dataset.')
            result_answer2 = messagebox.askyesno("Number recognition",
                                                 "Would you like to re-train model with new item?")

            if result_answer2:
                network1.train_neural_network()

    def train_model(self):
        messagebox.showinfo("Training", "This might take few minutes, please wait.")
        result = network1.train_neural_network()
        if result == 1:
            messagebox.showinfo("Training", "Training completed!")
        else:
            print("Something is no yes")

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
        filemenu = Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Save', command=self.save)
        optionmenu = Menu(menu, tearoff=0)
        menu.add_cascade(label='Options', menu=optionmenu)
        optionmenu.add_command(label='Clear canvas', command=self.clear)
        optionmenu.add_command(label='Recognize digit', command=self.recognize_number)
        optionmenu.add_command(label='Train neural network', command=self.train_model)
        optionmenu.add_command(label='Exit', command=self.master.destroy)


class InputBox:
    def __init__(self, text=""):
        self.root1 = Tk()
        self.get = ""
        self.root1.geometry("300x80")
        self.root1.title("Number?")
        self.label_file_name = Label(self.root1, text=text)
        self.label_file_name.pack()
        self.entry = Entry(self.root1)
        self.entry.pack()
        self.entry.focus()
        self.entry.bind("<Return>", lambda x: self.getinput(self.entry.get()))
        self.root1.mainloop()

    def getinput(self, value):
        self.get = value
        self.root1.destroy()
        self.root1.quit()


if __name__ == '__main__':
    network1 = NeuralNetwork()
    files_list = listdir('working_dataset')
    if files_list.__len__() == 4:
        print("Loading working dataset...")
        network1.dataset = network1.load_working_dataset()
        print("Ok!")
    else:
        print("Loading clean dataset...")
        network1.dataset = network1.load_clean_dataset()
        print("Ok!")

    root = Tk()
    Main(root)
    root.title('DrawingApp')
    root.mainloop()
