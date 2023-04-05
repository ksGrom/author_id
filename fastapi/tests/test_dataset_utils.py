import pytest
import dataset_utils as u


@pytest.mark.parametrize(
    ['orig', 'res'], [
        ('Привет...', 'Привет…'),
        ('музей-усадьба «Ясная Поляна»', 'музей-усадьба "Ясная Поляна"'),
        ('Текст.\nТекст.', 'Текст. Текст.'),
    ]
)
def test_replace_punctuation_marks(orig, res):
    assert res == u._replace_punctuation_marks(orig)
