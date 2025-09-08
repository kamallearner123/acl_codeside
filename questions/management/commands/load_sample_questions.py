from django.core.management.base import BaseCommand
from django.utils.text import slugify
from questions.models import Question, Tag


class Command(BaseCommand):
    help = 'Load sample questions into the database'

    def handle(self, *args, **options):
        # Create tags
        tags_data = [
            'Array', 'String', 'Hash Table', 'Dynamic Programming',
            'Math', 'Two Pointers', 'Greedy', 'Sorting', 'Tree',
            'Binary Search', 'Recursion', 'Stack', 'Queue'
        ]
        
        tags = {}
        for tag_name in tags_data:
            tag, created = Tag.objects.get_or_create(name=tag_name)
            tags[tag_name] = tag
            if created:
                self.stdout.write(f'Created tag: {tag_name}')

        # Sample questions
        questions_data = [
            {
                'title': 'Two Sum',
                'difficulty': 'easy',
                'description': '''Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.''',
                'example_input': '[2,7,11,15], target = 9',
                'example_output': '[0,1]',
                'constraints': '''- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.''',
                'hints': '''1. Try using a hash table to store the numbers you've seen.
2. For each number, check if (target - number) exists in the hash table.''',
                'template_code': '''def two_sum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '[2,7,11,15], 9', 'expected': '[0,1]'},
                    {'input': '[3,2,4], 6', 'expected': '[1,2]'},
                    {'input': '[3,3], 6', 'expected': '[0,1]'}
                ],
                'tags': ['Array', 'Hash Table']
            },
            {
                'title': 'Reverse String',
                'difficulty': 'easy',
                'description': '''Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.''',
                'example_input': '["h","e","l","l","o"]',
                'example_output': '["o","l","l","e","h"]',
                'constraints': '''- 1 <= s.length <= 10^5
- s[i] is a printable ascii character.''',
                'hints': '''1. Use two pointers approach.
2. Swap characters from both ends moving towards the center.''',
                'template_code': '''def reverse_string(s):
    """
    :type s: List[str]
    :rtype: None Do not return anything, modify s in-place instead.
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '["h","e","l","l","o"]', 'expected': '["o","l","l","e","h"]'},
                    {'input': '["H","a","n","n","a","h"]', 'expected': '["h","a","n","n","a","H"]'}
                ],
                'tags': ['String', 'Two Pointers']
            },
            {
                'title': 'Valid Parentheses',
                'difficulty': 'easy',
                'description': '''Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.''',
                'example_input': '"()[]{}"',
                'example_output': 'true',
                'constraints': '''- 1 <= s.length <= 10^4
- s consists of parentheses only '()[]{}'.''',
                'hints': '''1. Use a stack data structure.
2. Push opening brackets onto the stack.
3. When you encounter a closing bracket, check if it matches the top of the stack.''',
                'template_code': '''def is_valid(s):
    """
    :type s: str
    :rtype: bool
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '"()"', 'expected': 'true'},
                    {'input': '"()[]{}"', 'expected': 'true'},
                    {'input': '"(]"', 'expected': 'false'}
                ],
                'tags': ['String', 'Stack']
            },
            {
                'title': 'Maximum Subarray',
                'difficulty': 'medium',
                'description': '''Given an integer array nums, find the subarray with the largest sum, and return its sum.

A subarray is a contiguous non-empty sequence of elements within an array.''',
                'example_input': '[-2,1,-3,4,-1,2,1,-5,4]',
                'example_output': '6',
                'constraints': '''- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4''',
                'hints': '''1. This is a classic dynamic programming problem.
2. At each position, decide whether to start a new subarray or extend the existing one.
3. Keep track of the maximum sum seen so far.''',
                'template_code': '''def max_subarray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '[-2,1,-3,4,-1,2,1,-5,4]', 'expected': '6'},
                    {'input': '[1]', 'expected': '1'},
                    {'input': '[5,4,-1,7,8]', 'expected': '23'}
                ],
                'tags': ['Array', 'Dynamic Programming']
            },
            {
                'title': 'Merge Two Sorted Lists',
                'difficulty': 'easy',
                'description': '''You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.''',
                'example_input': 'list1 = [1,2,4], list2 = [1,3,4]',
                'example_output': '[1,1,2,3,4,4]',
                'constraints': '''- The number of nodes in both lists is in the range [0, 50].
- -100 <= Node.val <= 100
- Both list1 and list2 are sorted in non-decreasing order.''',
                'hints': '''1. Use a dummy node to simplify the logic.
2. Compare the values at the current nodes and choose the smaller one.
3. Move the pointer of the chosen list forward.''',
                'template_code': '''# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_two_lists(list1, list2):
    """
    :type list1: Optional[ListNode]
    :type list2: Optional[ListNode]
    :rtype: Optional[ListNode]
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '[1,2,4], [1,3,4]', 'expected': '[1,1,2,3,4,4]'},
                    {'input': '[], []', 'expected': '[]'},
                    {'input': '[], [0]', 'expected': '[0]'}
                ],
                'tags': ['Recursion']
            },
            {
                'title': 'Climbing Stairs',
                'difficulty': 'easy',
                'description': '''You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?''',
                'example_input': '3',
                'example_output': '3',
                'constraints': '''- 1 <= n <= 45''',
                'hints': '''1. This is a Fibonacci sequence problem.
2. To reach step n, you can either come from step n-1 or step n-2.
3. Use dynamic programming to avoid redundant calculations.''',
                'template_code': '''def climb_stairs(n):
    """
    :type n: int
    :rtype: int
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '2', 'expected': '2'},
                    {'input': '3', 'expected': '3'},
                    {'input': '4', 'expected': '5'}
                ],
                'tags': ['Math', 'Dynamic Programming']
            },
            {
                'title': 'Binary Tree Inorder Traversal',
                'difficulty': 'easy',
                'description': '''Given the root of a binary tree, return the inorder traversal of its nodes' values.

Inorder traversal visits nodes in the order: Left, Root, Right.''',
                'example_input': 'root = [1,null,2,3]',
                'example_output': '[1,3,2]',
                'constraints': '''- The number of nodes in the tree is in the range [0, 100].
- -100 <= Node.val <= 100''',
                'hints': '''1. Use recursion: visit left subtree, current node, then right subtree.
2. Alternatively, use an iterative approach with a stack.''',
                'template_code': '''# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '[1,null,2,3]', 'expected': '[1,3,2]'},
                    {'input': '[]', 'expected': '[]'},
                    {'input': '[1]', 'expected': '[1]'}
                ],
                'tags': ['Tree', 'Stack', 'Recursion']
            },
            {
                'title': 'Contains Duplicate',
                'difficulty': 'easy',
                'description': '''Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.''',
                'example_input': '[1,2,3,1]',
                'example_output': 'true',
                'constraints': '''- 1 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9''',
                'hints': '''1. Use a hash set to keep track of seen numbers.
2. If you encounter a number you've seen before, return true.''',
                'template_code': '''def contains_duplicate(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    # Write your code here
    pass''',
                'test_cases': [
                    {'input': '[1,2,3,1]', 'expected': 'true'},
                    {'input': '[1,2,3,4]', 'expected': 'false'},
                    {'input': '[1,1,1,3,3,4,3,2,4,2]', 'expected': 'true'}
                ],
                'tags': ['Array', 'Hash Table', 'Sorting']
            }
        ]

        for question_data in questions_data:
            slug = slugify(question_data['title'])
            question, created = Question.objects.get_or_create(
                slug=slug,
                defaults={
                    'title': question_data['title'],
                    'description': question_data['description'],
                    'difficulty': question_data['difficulty'],
                    'example_input': question_data['example_input'],
                    'example_output': question_data['example_output'],
                    'constraints': question_data['constraints'],
                    'hints': question_data['hints'],
                    'template_code': question_data['template_code'],
                    'test_cases': question_data['test_cases']
                }
            )
            
            if created:
                # Add tags
                for tag_name in question_data['tags']:
                    question.tags.add(tags[tag_name])
                
                self.stdout.write(
                    self.style.SUCCESS(f'Created question: {question.title}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Question already exists: {question.title}')
                )

        self.stdout.write(
            self.style.SUCCESS('Successfully loaded sample questions!')
        )
