

OBJECT_IDENTIFICATION = '''
You see a gripper of a robot holding an object. What is the color 
of the object? Just give a one word answer mentioning the color.
'''

SHELF_DESCRIPTION = '''
There are 3 shelves in your field of view. The top
shelf is labelld 1, the middle shelf is labeled 2, and 
the bottom shelf is labeled 3. There are boxes of 
different colors. For all the boxes in your view that are on the shelves
give me the color of the block and the shelf that it is on.
Give the result in a json format with the following structure:
[
    {
        "color": "red",
        "shelf": 1
    },
    {
        "color": "blue",
        "shelf": 2
    }
]
'''
            