from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='versterken',
      url='https://github.com/katabaticwind/versterken',
      version='0.1',
      description='Deep Reinforcement Learning with TensorFlow',
      long_description=readme,
      license=license,
      packages=['main']
)
