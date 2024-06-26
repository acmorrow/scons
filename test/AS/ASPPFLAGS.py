#!/usr/bin/env python
#
# MIT License
#
# Copyright The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys

import TestSCons

_python_ = TestSCons._python_

test = TestSCons.TestSCons()
_exe = TestSCons._exe

test.file_fixture('mylink.py')
test.file_fixture(['fixture', 'myas_args.py'])

o = ' -x'
o_c = ' -x -c'

test.write('SConstruct', """
DefaultEnvironment(tools=[])
env = Environment(tools=['link','as','cc'],
                  LINK = r'%(_python_)s mylink.py',
                  LINKFLAGS = [],
                  ASPP = r'%(_python_)s myas_args.py', ASPPFLAGS = '-x',
                  CC = r'%(_python_)s myas_args.py')
env.Program(target = 'test1', source = 'test1.spp')
env.Program(target = 'test2', source = 'test2.SPP')
""" % locals())

test.write('test1.spp', r"""This is a .spp file.
#as
#link
""")

test.write('test2.SPP', r"""This is a .SPP file.
#as
#link
""")

test.run(arguments = '.', stderr = None)

test.must_match('test1' + _exe, "%s\nThis is a .spp file.\n" % o_c)
test.must_match('test2' + _exe, "%s\nThis is a .SPP file.\n" % o_c)

test.pass_test()

# Local Variables:
# tab-width:4
# indent-tabs-mode:nil
# End:
# vim: set expandtab tabstop=4 shiftwidth=4:
