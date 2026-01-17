#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import os
picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')

libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import logging
from waveshare_epd import epd7in5_V2
import time
from PIL import Image,ImageDraw,ImageFont
import traceback

# Initialize the E-Paper display and fonts
epd = epd7in5_V2.EPD()

font24 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 24)
font18 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 18)
font35 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 35)
font12 = ImageFont.truetype(os.path.join(picdir, 'Font.ttc'), 12, index=1)
Himage = Image.new('1', (epd.height, epd.width), 255)  # 255: clear the frame
draw = ImageDraw.Draw(Himage)

def set_stage(emotion):
    """
    Sets the background and character image on the e-paper display.
    """
    
    epd.init()
    epd.Clear()

    # Load and paste the stage and character images
    stage_bmp = Image.open(os.path.join(picdir, 'BrambleStage.bmp'))
    Himage.paste(stage_bmp, (0, 0))

    character_bmp = Image.open(os.path.join(picdir, 'Bramble' + emotion + '.bmp'))
    Himage.paste(character_bmp, (90, 0))

    blank_stage_bmp = Image.open(os.path.join(picdir, 'BlankStage.bmp'))
    Himage.paste(blank_stage_bmp, (0, 300))

    epd.display(epd.getbuffer(Himage))

def render_paragraph(long_text):
    """
    Renders a long string of text onto the e-paper display with word wrapping.
    """
    words = long_text.split()
    current_line = ""
    line_count = 0
    max_lines = 12
    max_width = 440

    epd.init_part()

    # Create a dummy drawing context to measure text width
    dummy_image = Image.new('RGB', (1, 1))
    draw_dummy = ImageDraw.Draw(dummy_image)

    for word in words:
        if line_count >= max_lines:
            break

        potential_line = current_line + (" " if current_line else "") + word
        potential_width = draw_dummy.textlength(potential_line, font=font24)

        if potential_width <= max_width:
            current_line = potential_line
        else:
            if current_line:
                draw.text((20, 300 + line_count * 28), current_line, font=font24, fill=0)
                epd.display_Partial(epd.getbuffer(Himage), 0, 0, epd.width, epd.height)
                line_count += 1
                if line_count >= max_lines:
                    break
            current_line = word

    # Render the last line if it exists
    if current_line and line_count < max_lines:
        draw.text((20, 300 + line_count * 28), current_line, font=font24, fill=0)
        epd.display_Partial(epd.getbuffer(Himage), 0, 0, epd.width, epd.height)

def main():
    """
    Main function to run the display sequence.
    """
    try:
        set_stage("Neutral")

        # Define example texts
        text_intro = "Hello, I'm Primer, your AI teacher in the form of a book, ready to help you explore the world of knowledge. My lessons are designed to simplify complex topics, making learning both clear and enjoyable. From math and science to history and beyond, I guide you through each subject. Let's embark on a journey of endless curiosity together!"
        text_joke = "A chicken walks into the library, clucks, 'Book!' and the librarian hands it a book. The next day, the same chicken comes back, says 'Book! Book!' and leaves with two. Curious, the librarian follows the chicken through the streets, into the woods, and finds it handing the books to a frog sitting on a lily pad. The frog looks at each book and says, 'Read it. Read it.'"
        text_FRC = "Great question! The FIRST Robotics Competition, or FRC for short, is a worldwide high school robotics competition where teams of students design, build, and program robots to compete in exciting games. But FRC isn't just about robots—it's also about teamwork, problem-solving, and learning real-world engineering skills. It's like a sport for the mind, where everyone can go pro in STEM!"
        text_tumbly = "Hey there, I'm Tumbly! I'm your curious elephant buddy who loves big ideas and even bigger questions. I might trip over my own feet sometimes, but that just means I'm always learning something new! Together, we’ll explore science, art, and everything in between—with lots of laughs and “aha!” moments along the way. I believe mistakes are just stepping stones to discovery. Ready to tumble into learning?"

        # Run the display sequence
        render_paragraph(text_intro)
        time.sleep(5)  # Add a delay for readability

        set_stage("Laughing")

        epd.init_part()
        draw.text((20, 300 + 1 * 28), "Why did the bicycle fall over?", font=font24, fill=0)
        epd.display_Partial(epd.getbuffer(Himage), 0, 0, epd.width, epd.height)

        time.sleep(2)

        draw.text((20, 300 + 4 * 28), "Because it was two tired!", font=font24, fill=0)
        epd.display_Partial(epd.getbuffer(Himage), 0, 0, epd.width, epd.height)

        time.sleep(10)

        epd.init()
        set_stage("Sleeping")

        epd.sleep()

    except IOError as e:
        logging.info(e)
    except KeyboardInterrupt:
        logging.info("ctrl + c:")
        epd7in5_V2.epdconfig.module_exit(cleanup=True)
        sys.exit()

if __name__ == '__main__':
    main()
