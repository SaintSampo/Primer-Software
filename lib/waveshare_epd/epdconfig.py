# /*****************************************************************************
# Using cdevGPIO now
# /*****************************************************************************
# * | File        :	  epdconfig.py
# * | Author      :   Waveshare team
# * | Function    :   Hardware underlying interface
# * | Info        :
# *----------------
# * | This version:   V1.2
# * | Date        :   2022-10-29
# * | Info        :   
# ******************************************************************************
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import os
import logging
import sys
import time
import subprocess

from ctypes import *

logger = logging.getLogger(__name__)

class RubikPi:
    GPIOCHIP = "/dev/gpiochip4"
    # Pin definition
    RST_PIN  = 8 #17
    DC_PIN   = 44 #25
    CS_PIN   = 55 #8
    BUSY_PIN = 27 #24
    PWR_PIN  = 101 #18
    MOSI_PIN = 49 #10
    SCLK_PIN = 50 #11
    
    def __init__(self):
        import spidev
        from periphery import GPIO
        
        # Open SPI device
        self.SPI = spidev.SpiDev()

        # Open GPIO pins
        self.GPIO_RST_PIN    = GPIO(self.GPIOCHIP, self.RST_PIN, "out")
        self.GPIO_DC_PIN     = GPIO(self.GPIOCHIP, self.DC_PIN, "out")
        # self.GPIO_CS_PIN     = GPIO(self.GPIOCHIP, self.CS_PIN, "out")
        self.GPIO_PWR_PIN    = GPIO(self.GPIOCHIP, self.PWR_PIN, "out")
        self.GPIO_BUSY_PIN   = GPIO(self.GPIOCHIP, self.BUSY_PIN, "in")
        

    def digital_write(self, pin, value):
        val = bool(value)
        if pin == self.RST_PIN:
            self.GPIO_RST_PIN.write(val)
        elif pin == self.DC_PIN:
            self.GPIO_DC_PIN.write(val)
        # elif pin == self.CS_PIN:
        #     self.GPIO_CS_PIN.write(val)
        elif pin == self.PWR_PIN:
            self.GPIO_PWR_PIN.write(val)

    def digital_read(self, pin):
        if pin == self.BUSY_PIN:
            return self.GPIO_BUSY_PIN.read()
        elif pin == self.RST_PIN:
            return self.GPIO_RST_PIN.read()
        elif pin == self.DC_PIN:
            return self.GPIO_DC_PIN.read()
        # elif pin == self.CS_PIN:
        #     return self.GPIO_CS_PIN.read()
        elif pin == self.PWR_PIN:
            return self.GPIO_PWR_PIN.read()

    def delay_ms(self, delaytime):
        time.sleep(delaytime / 1000.0)

    def spi_writebyte(self, data):
        self.SPI.writebytes(data)

    def spi_writebyte2(self, data):
        self.SPI.writebytes2(data)

    def DEV_SPI_write(self, data):
        self.DEV_SPI.DEV_SPI_SendData(data)

    def DEV_SPI_nwrite(self, data):
        self.DEV_SPI.DEV_SPI_SendnData(data)

    def DEV_SPI_read(self):
        return self.DEV_SPI.DEV_SPI_ReadData()

    def module_init(self, cleanup=False):
        self.GPIO_PWR_PIN.write(True) # Set power pin to HIGH
        
        if cleanup:
            find_dirs = [
                os.path.dirname(os.path.realpath(__file__)),
                '/usr/local/lib',
                '/usr/lib',
            ]
            self.DEV_SPI = None
            for find_dir in find_dirs:
                val = int(os.popen('getconf LONG_BIT').read())
                logging.debug("System is %d bit"%val)
                if val == 64:
                    so_filename = os.path.join(find_dir, 'DEV_Config_64.so')
                else:
                    so_filename = os.path.join(find_dir, 'DEV_Config_32.so')
                if os.path.exists(so_filename):
                    self.DEV_SPI = CDLL(so_filename)
                    break
            if self.DEV_SPI is None:
                RuntimeError('Cannot find DEV_Config.so')

            self.DEV_SPI.DEV_Module_Init()

        else:
            # SPI device, bus = 0, device = 0
            self.SPI.open(12, 0)
            self.SPI.max_speed_hz = 4000000
            self.SPI.mode = 0b00
        return 0

    def module_exit(self, cleanup=False):
        logger.debug("spi end")
        self.SPI.close()

        self.GPIO_RST_PIN.write(False)
        self.GPIO_DC_PIN.write(False)
        self.GPIO_PWR_PIN.write(False)
        logger.debug("close 5V, Module enters 0 power consumption ...")
        
        if cleanup:
            self.GPIO_RST_PIN.close()
            self.GPIO_DC_PIN.close()
            # self.GPIO_CS_PIN.close()
            self.GPIO_PWR_PIN.close()
            self.GPIO_BUSY_PIN.close()

implementation = RubikPi()

for func in [x for x in dir(implementation) if not x.startswith('_')]:
    setattr(sys.modules[__name__], func, getattr(implementation, func))

### END OF FILE ###