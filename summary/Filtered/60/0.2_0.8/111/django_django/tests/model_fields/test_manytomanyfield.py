from django.apps import apps
from django.db import models
from django.test import SimpleTestCase, TestCase
from django.test.utils import isolate_apps

from .models import ManyToMany


class ManyToManyFieldTests(SimpleTestCase):
    def test_abstract_model_pending_operations(self):
        """
        Many-to-many fields declared on abstract models should not add lazy
        relations to resolve relationship declared as string (#24215).
        """
        pending_ops_before = list(apps._pending_operations.items())

        class AbstractManyToManyModel(models.Model):
            fk = models.ForeignKey("missing.FK", models.CASCADE)

            class Meta:
                abstract = True

        self.assertIs(AbstractManyToManyModel._meta.apps, apps)
        self.assertEqual(
            pending_ops_before,
            list(apps._pending_operations.items()),
            "Pending lookup added for a many-to-many field on an abstract model",
        )

    @isolate_apps("model_fields", "model_fields.tests")
    def test_abstract_model_app_relative_foreign_key(self):
        """
        Tests the behavior of an abstract model with an app-relative foreign key.
        
        This function checks that when an abstract model with a ManyToManyField through a related model is defined, and the related model and through model reference an app label, the correct models are resolved and linked.
        
        Parameters:
        None
        
        Keywords:
        None
        
        Returns:
        None
        
        Explanation:
        The function defines an abstract model `AbstractReferent` with a ManyToManyField to another model `Referred` through a `Through`
        """

        class AbstractReferent(models.Model):
            reference = models.ManyToManyField("Referred", through="Through")

            class Meta:
                app_label = "model_fields"
                abstract = True

        def assert_app_model_resolved(label):
            """
            Assert that the model referenced by the 'reference' field in the `ConcreteReferent` model is correctly resolved.
            
            Args:
            label (str): The app label used for the model definitions.
            
            This function creates three models: `Referred`, `Through`, and `ConcreteReferent`. The `ConcreteReferent` model has a ForeignKey field named 'reference' that references the `Referred` model. The function then checks two things:
            1. The `related_model` attribute of the '
            """

            class Referred(models.Model):
                class Meta:
                    app_label = label

            class Through(models.Model):
                referred = models.ForeignKey("Referred", on_delete=models.CASCADE)
                referent = models.ForeignKey(
                    "ConcreteReferent", on_delete=models.CASCADE
                )

                class Meta:
                    app_label = label

            class ConcreteReferent(AbstractReferent):
                class Meta:
                    app_label = label

            self.assertEqual(
                ConcreteReferent._meta.get_field("reference").related_model, Referred
            )
            self.assertEqual(ConcreteReferent.reference.through, Through)

        assert_app_model_resolved("model_fields")
        assert_app_model_resolved("tests")

    def test_invalid_to_parameter(self):
        msg = (
            "ManyToManyField(1) is invalid. First parameter to "
            "ManyToManyField must be either a model, a model name, or the "
            "string 'self'"
        )
        with self.assertRaisesMessage(TypeError, msg):

            class MyModel(models.Model):
                m2m = models.ManyToManyField(1)

    @isolate_apps("model_fields")
    def test_through_db_table_mutually_exclusive(self):
        class Child(models.Model):
            pass

        class Through(models.Model):
            referred = models.ForeignKey(Child, on_delete=models.CASCADE)
            referent = models.ForeignKey(Child, on_delete=models.CASCADE)

        msg = "Cannot specify a db_table if an intermediary model is used."
        with self.assertRaisesMessage(ValueError, msg):

            class MyModel(models.Model):
                m2m = models.ManyToManyField(
                    Child,
                    through="Through",
                    db_table="custom_name",
                )


class ManyToManyFieldDBTests(TestCase):
    def test_value_from_object_instance_without_pk(self):
        obj = ManyToMany()
        self.assertEqual(obj._meta.get_field("m2m").value_from_object(obj), [])

    def test_value_from_object_instance_with_pk(self):
        obj = ManyToMany.objects.create()
        related_obj = ManyToMany.objects.create()
        obj.m2m.add(related_obj)
        self.assertEqual(
            obj._meta.get_field("m2m").value_from_object(obj), [related_obj]
        )
