# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:00:38 2015
This file is part of pyNLO.

    pyNLO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pyNLO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pyNLO.  If not, see <http://www.gnu.org/licenses/>.
@author: ycasg
"""

import jsonpickle
import os

class JSONFiberLoader:
    """ Load fiber parameters from pickle file. """
    fiber_names = None
    def __init__(self, fiber_collection="general_fibers", file_dir = None):
        """ Initialize by reading pickles fiber parameters. If you have a pickle
        containing your own fiber types, change general_fibers to your own
        (.pickle will be appended.)"""
        if file_dir is None:
            root = os.path.abspath(os.path.dirname(__file__))
        else:
            root = file_dir
        picklefile = os.path.join(root, fiber_collection+'.txt')
        file_handle =  open(picklefile, 'r')
        data= file_handle.read()
        self.fibers = jsonpickle.decode(data)
        file_handle.close()
    def print_fiber_list(self):
        """ Print list of all fibers in database. """
        self.fiber_names = []
        for each in self.fibers.keys():
            print 'fiber: ',each
            self.fiber_names.append(each)
    def get_fiber(self, name):
        """ Retrieve fiber parameters for fiber "name" """
        fiberspecs = self.fibers[name]
        return fiberspecs
        