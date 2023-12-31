# Проект Определение авторства текстов (МОВС 2022)

**Прим.** В этом репозитории я оставил только свой код — микросервис на FastAPI (полностью рабочий).
Более подробный readme — в папке `fastapi`.

***Команда:***

1) Громов Кирилл
2) Казачкова Анна
3) Вересников Артем

***Цель проекта:*** создать модель классификации текстов по авторству

### Результаты
- На наборе произведений 9 русских писателей XIX-XX веков исследованы
различные подходы, основанные на машинном обучении, для классификации текстов по авторству.
Выбран оптимальный алгоритм предобработки данных и обучения модели.
- Разработан микросервис на основе фреймворка FastAPI.
Сервис позволяет хранить и обрабатывать тестовые и тренировочные 
наборы текстов; обучать, дообучать, тестировать и применять 
модели машинного обучения для классификации текстов по авторству; 
формировать из txt-файлов наборы данных в подходящем для работы с моделями формате. 
Код сервиса и документация расположены в каталоге `fastapi`.
