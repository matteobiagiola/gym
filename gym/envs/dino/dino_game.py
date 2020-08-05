
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from PIL import Image
import numpy as np
import base64
from io import BytesIO

init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

class DinoGame:
    def __init__(self, chromebrowser_path, render=False, accelerate=False, autoscale=False):
        game_url = "chrome://dino"
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument('--no-sandbox')
        self.browser = webdriver.Chrome(executable_path=chromebrowser_path, options=chrome_options)
        self.browser.set_window_position(x=-10,y=0)
        self.browser.get(game_url)
        self.browser.execute_script(init_script)
        self.browser.implicitly_wait(30)
        self.browser.maximize_window()
        if not accelerate:
            self.set_parameter('config.ACCELERATION', 0)
        if not autoscale:
            self.browser.execute_script('Runner.instance_.setArcadeModeContainerScale = function(){};')

        self.start_game()

    def start_game(self):
        while not self.is_playing():
            self.press_space()

    def get_image(self):
        image_b64 = self.get_canvas()
        image = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        return image

    def press_space(self):
        return self.browser.find_element_by_tag_name('body').send_keys(Keys.SPACE)

    def is_crashed(self):
        return self.browser.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self.browser.execute_script("return Runner.instance_.playing")

    def restart(self):
        self.browser.execute_script("Runner.instance_.restart()")
        
    def press_up(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN) 

    def press_right(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_RIGHT)

    def get_score(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def get_highscore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.highScore")
        for i in range(len(score_array)):
            if score_array[i] == '':
                break
        score_array = score_array[i:]        
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.browser.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self.browser.execute_script("return Runner.instance_.play()")

    def get_canvas(self):
        return self.browser.execute_script('return document.getElementsByClassName("runner-canvas")[0].toDataURL().substring(22);')
    
    def set_parameter(self, key, value):
        self.browser.execute_script('Runner.{} = {};'.format(key, value))
    
    def restore_parameter(self, key):
        self.set_parameter(self, key, self.defaults[key])

    def end(self):
        self.browser.quit()