import tkinter as tk


class GalaxyApp(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.screenTitle = tk.Label(self, text="Galaxy Classifier GUI").grid(row=1, column=1)


if __name__ == '__main__':
    galaxyApp = GalaxyApp()
    galaxyApp.title('GUI')

    galaxyApp.mainloop()
