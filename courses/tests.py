from django.test import TestCase
import re

from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import Course, GroupFeedback


class CoursePageTests(TestCase):
    def setUp(self):
        self.embedded_course = Course.objects.create(
            title='IoT Devices',
            slug='iot-devices',
        )
        self.ai_course = Course.objects.create(
            title='Agentic AI',
            slug='agentic-ai',
        )
        self.security_course = Course.objects.create(
            title='Networking and Security',
            slug='networking-security',
        )
        self.other_course = Course.objects.create(
            title='New Training Program',
            slug='new-training-program',
        )
        self.group_courses = [
            Course.objects.get_or_create(
                slug=slug,
                defaults={'title': title},
            )[0]
            for slug, title in (
                ('arm-cortex-m-architecture', 'ARM Cortex-M Architecture for Embedded Engineers'),
                ('stm32-firmware-development-with-c', 'STM32 Firmware Development with C'),
                ('embedded-rust-with-stm32', 'Embedded Rust with STM32'),
                ('visualise-stm32', 'Visualise STM32'),
            )
        ]

    def test_course_list_groups_courses_and_keeps_unmapped_courses_visible(self):
        response = self.client.get(reverse('courses:list'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Embedded Systems and IoT')
        self.assertContains(response, 'AI/ML and Agentic AI')
        self.assertContains(response, 'Cyber security')
        self.assertContains(response, reverse('courses:stm32_automotive_group'))
        groups = {
            group['title']: group['courses']
            for group in response.context['course_groups']
        }
        self.assertIn(self.embedded_course, groups['Embedded Systems and IoT'])
        self.assertIn(self.ai_course, groups['AI/ML and Agentic AI'])
        self.assertIn(self.security_course, groups['Cyber security'])
        self.assertIn(self.other_course, groups['Other Courses'])
        self.assertEqual(
            set(groups['STM32+C+Rust+Autmotive']),
            set(self.group_courses),
        )
        self.assertEqual(
            sum(len(courses) for courses in groups.values()),
            len(response.context['courses']),
        )

    def test_stm32_group_hub_lists_courses_setup_projects_and_feedback(self):
        response = self.client.get(reverse('courses:stm32_automotive_group'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'STM32+C+Rust+Autmotive')
        self.assertContains(response, 'ARM Cortex-M Architecture for Embedded Engineers')
        self.assertContains(response, 'Programming STM32 with C')
        self.assertContains(response, 'Embedded Rust with STM32')
        self.assertContains(response, 'Visualise STM32')
        self.assertContains(response, 'NUCLEO-F446RE')
        self.assertContains(response, 'macOS')
        self.assertContains(response, 'Linux')
        self.assertContains(response, 'Windows')
        self.assertContains(response, 'xcode-select --install')
        self.assertContains(response, 'sudo apt install -y git build-essential')
        self.assertContains(response, 'rustup target list --installed')
        self.assertContains(response, 'probe-rs list')
        self.assertContains(response, 'code --version')
        self.assertContains(response, 'Build Finished')
        self.assertContains(response, 'STM32CubeIDE includes the STM32CubeMX')
        self.assertContains(response, 'STM32CubeMX</a> separately')
        self.assertContains(response, 'stm32cubemx-home-check.png')
        self.assertContains(response, 'stm32cubeide-editor-check.png')
        self.assertContains(response, 'Screenshots: what to check after installing the STM32 tools')
        self.assertContains(response, 'Automotive cybersecurity')
        self.assertContains(response, 'Zephyr + industrial automation')
        self.assertContains(response, 'Email me a verification code')

    @override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
    def test_group_feedback_is_saved_only_after_email_verification(self):
        response = self.client.post(
            reverse('courses:stm32_group_feedback'),
            {
                'name': 'Test Learner',
                'email': 'learner@example.com',
                'course': 'Programming STM32 with C',
                'topic': GroupFeedback.Topic.SUGGESTION,
                'message': 'Please add a simulated CAN gateway project.',
            },
        )

        self.assertRedirects(response, reverse('courses:stm32_group_feedback_verify'))
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(GroupFeedback.objects.count(), 0)
        code = re.search(r'\b(\d{6})\b', mail.outbox[0].body).group(1)
        response = self.client.post(
            reverse('courses:stm32_group_feedback_verify'),
            {'code': code},
        )

        self.assertRedirects(response, reverse('courses:stm32_automotive_group'))
        feedback = GroupFeedback.objects.get()
        self.assertEqual(feedback.email, 'learner@example.com')
        self.assertEqual(feedback.course, 'Programming STM32 with C')
        self.assertEqual(feedback.message, 'Please add a simulated CAN gateway project.')
        self.assertEqual(len(mail.outbox), 3)
        self.assertEqual(mail.outbox[1].to, ['kamal@aptcomputinglabs.com'])
        self.assertIn('learner@example.com (verified)', mail.outbox[1].body)
        self.assertEqual(mail.outbox[2].to, ['learner@example.com'])

    def test_course_detail_offers_a_printable_manual(self):
        response = self.client.get(
            reverse('courses:detail', args=[self.embedded_course.slug])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Print Manual / Save as PDF')
        self.assertContains(response, 'window.print()')

    def test_visualise_stm32_course_links_to_interactive_book(self):
        course = Course.objects.get(slug='visualise-stm32')
        response = self.client.get(reverse('courses:detail', args=[course.slug]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Open Interactive Visualiser')
        self.assertContains(response, '22 step-by-step flows')
        self.assertContains(response, 'persistent, clickable STM32L476RG Cortex-M4 block')
        self.assertContains(
            response,
            reverse('courses:visualise_stm32_book_index'),
        )

    def test_visualise_stm32_book_and_assets_are_served(self):
        book_response = self.client.get(
            reverse('courses:visualise_stm32_book_index')
        )
        css_response = self.client.get(
            reverse(
                'courses:visualise_stm32_book_serve',
                args=['style.css'],
            )
        )
        js_response = self.client.get(
            reverse(
                'courses:visualise_stm32_book_serve',
                args=['app.js'],
            )
        )

        self.assertEqual(book_response.status_code, 200)
        book_content = b''.join(book_response.streaming_content).decode()
        self.assertIn('Visualise STM32', book_content)
        self.assertIn('01 · MASTER ARCHITECTURE', book_content)
        self.assertIn('id="flow-prompt-panel"', book_content)
        self.assertIn('id="flow-prompt-title"', book_content)
        self.assertIn('architecture-diagram', book_content)
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(js_response.status_code, 200)
        css_content = b''.join(css_response.streaming_content).decode()
        self.assertIn('.bus-wire.is-context { stroke-width: 7', css_content)
        self.assertIn('.diagram-edge.is-current { stroke:', css_content)
        self.assertIn('stroke-width: 6', css_content)
        js_content = b''.join(js_response.streaming_content).decode()
        self.assertIn('Power-on and reset', js_content)
        self.assertIn('AHB bus matrix', js_content)
        self.assertIn('instruction address + fetch control', js_content)
        self.assertIn('read data', js_content)
        self.assertIn('write data', js_content)
        self.assertIn('must have 22 detailed flows', js_content)
        self.assertIn('22 detailed flows', js_content)
        self.assertIn('Show prompt for flow', js_content)
        self.assertIn('renderFlowPrompt', js_content)
        self.assertIn('Power-on and reset', js_content)
