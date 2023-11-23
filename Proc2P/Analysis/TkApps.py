from tkinter import *
from tkinter import filedialog, messagebox, ttk
import os
import numpy


class App:
    def __init__(self, title='', root=None):
        if root is None:
            root = Tk()
            root.title(title)
            root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.destroy_on_quit = True
        else:
            self.destroy_on_quit = False
        self.root = root
        self.current_row = 0
        self.ret = 0

    def on_closing(self):
        if messagebox.askokcancel("Quit", "OK to quit?"):
            self.root.destroy()

    def row(self):
        self.current_row += 1
        return self.current_row - 1

    def loop(self):
        self.root.mainloop()

# class ListenKeys:
#     def __init__(self, kdict):
#         self.root = Tk()


# class Progress(App):
#     def __init(self, length):
#         super().__init__('Processing cells')
#         self.pb = ttk.Progressbar(self.root, orient=HORIZONTAL, mode="indeterminate", maximum=length, value=0)
#         self.pb.grid(row=0, column=0, sticky=N + W)
#
#     def set(self, n):
#         self.pb['value'] = n


class ShowMessage(App):
    def __init__(self, title, root=None, msg=''):
        # super().__init__(title, root)
        messagebox.showinfo(title, msg)
        # self.loop()


class EntryDialog(App):
    def __init__(self, title, root=None):
        super().__init__(title, root)
        self.val = StringVar()

        self.w = Toplevel(self.root)
        if self.destroy_on_quit:
            self.root.withdraw()

        self.frame = Frame(self.w)
        self.frame.grid(row=0, column=0, sticky=N + W)

        Label(self.frame, text=title).grid(row=self.row())
        Entry(self.frame, textvariable=self.val, width=20).grid(row=self.row())
        Button(self.frame, text='OK', command=self.ok_callback).grid(row=self.row())
        self.w.bind('<Return>', self.ok_callback)
        self.loop()

    def ok_callback(self, *args):
        self.ret = self.val.get()
        self.w.quit()
        if self.destroy_on_quit:
            self.root.destroy()

    def kill(self):
        self.w.destroy()

class PickFromList(App):
    def __init__(self, keys, root=None, multiple=False):
        super().__init__('Select', root)
        self.multiple = multiple
        self.val = IntVar()

        if type(keys) == int:
            keys = list(range(keys))

        self.w = Toplevel(self.root)
        if self.destroy_on_quit:
            self.root.withdraw()

        self.frame = Frame(self.w)
        self.frame.grid(row=0, column=0, sticky=N + W)

        if multiple:
            selectmode = EXTENDED
        else:
            selectmode = SINGLE
        self.key = Listbox(self.frame, selectmode=selectmode, exportselection=0)
        self.key.grid(row=0, column=0)
        width = 20
        for key in keys:
            key = str(key)
            self.key.insert(END, key)
            width = max(width, len(key))
        self.key.config(width=width, height=min(50,len(keys)))
        Button(self.frame, text='OK', command=self.ok_callback).grid(row=1, column=0)
        if multiple:
            self.key.select_set(0, END)
        self.w.bind('<Double-Button-1>', self.ok_callback)
        self.loop()

    def ok_callback(self, *args):
        if self.multiple:
            self.ret = self.key.curselection()
        else:
            self.ret = int(self.key.curselection()[0])
            self.val.set(self.ret)
        self.w.quit()
        if self.destroy_on_quit:
            self.root.destroy()

    def kill(self):
        self.w.destroy()

class SourceTarget(App):
    def __init__(self, path):
        super().__init__('Select source and target')
        self.path = path

        w = Toplevel(self.root)
        if self.destroy_on_quit:
            self.root.withdraw()

        self.frame = Frame(w)
        self.frame.grid(row=0, column=0, sticky=N + W)

        Label(self.frame, text='Source').grid(row=0, column=0)
        self.source = Listbox(self.frame, selectmode=SINGLE, exportselection=0)
        self.source.grid(row=1, column=0)

        Label(self.frame, text='Target').grid(row=0, column=1)
        self.target = Listbox(self.frame, selectmode=SINGLE, exportselection=0)
        self.target.grid(row=1, column=1)

        self.getfiles()

        Button(self.frame, text='OK', command=self.ok_callback).grid(row=2, column=1, sticky=S+E)
        Button(self.frame, text='Cancel', command=self.root.destroy).grid(row=2, column=1, sticky=S+W)

        self.loop()

    def ok_callback(self):
        self.ret = self.source.get(self.source.curselection()), self.target.get(self.target.curselection())
        if self.destroy_on_quit:
            self.root.destroy()

    def getfiles(self):
        pfs=[]
        for f in os.listdir(self.path):
            if f.endswith('.sbx'):
                if f.endswith('_nonrigid.sbx'):
                    prefix = f[:f.find('_nonrigid')]
                else:
                    prefix = f[:-4]
                if prefix not in pfs:
                    pfs.append(prefix)
        pfs.sort()
        for prefix in pfs:
            self.source.insert(END, prefix)
            self.target.insert(END, prefix)
        wlen = min(len(pfs), 30)
        self.source.config(height=wlen)
        self.target.config(height=wlen)


    # fn = filedialog.askopenfilename(initialdir=oloc)

#
# if __name__ == '__main__':
    # a=Progress(100)
    # print(SourceTarget('E://2pdata//camkii_2//').ret)
    # print(PickFromList(10).ret)
