from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class Applicant:
    school: fields.Boolean = fields.Boolean(
        title="School",
        description=
        "Student's school (true - Gabriel Pereira, false - Mousinho da Silveira)"
    )
    sex: fields.Boolean = fields.Boolean(
        title="Sex", description="Student's sex (true - female, false - male)")
    age: fields.Integer = fields.Integer(title="Age",
                                         description="Student's age",
                                         min=15,
                                         max=22)
    address: fields.Boolean = fields.Boolean(
        title="Address",
        description="Student's home address type (true - urban, false - rural)"
    )
    family_size: fields.Boolean = fields.Boolean(
        title="Family size",
        description="Family size (true - <=3, false - >3)")
    p_status: fields.Boolean = fields.Boolean(
        title="P status",
        description=
        "Parent's cohabitation status (true - living together, false - apart)")
    mother_edu: fields.Integer = fields.Integer(
        title="Mother's education",
        description=
        "Mother's education level (0 - none, 1 - 4th grade, 2 - 5~9th grade, 3 - secondary, 4 - higher)",
        min=0,
        max=4)
    father_edu: fields.Integer = fields.Integer(
        title="Father's education",
        description=
        "Father's education level (0 - none, 1 - 4th grade, 2 - 5~9th grade, 3 - secondary, 4 - higher)",
        min=0,
        max=4)
    mother_job: fields.Integer = fields.Integer(
        title="Mother's job",
        description=
        "Mother's job (0 - teacher, 1 - health-related, 2 - civil services, 3 - at home, 4 - other)",
        min=0,
        max=4)
    father_job: fields.Integer = fields.Integer(
        title="Father's job",
        description=
        "Father's job (0 - teacher, 1 - health-related, 2 - civil services, 3 - at home, 4 - other)",
        min=0,
        max=4)
    reason: fields.Integer = fields.Integer(
        title="Reason",
        description=
        "Reason to choose this school (0 - close to home, 1 - school reputation, 2 - course preference, 3 - other)",
        min=0,
        max=3)
    guardian: fields.Integer = fields.Integer(
        title="Guardian",
        description="Student's guardian (0 - mother, 1 - father, 2 - other)",
        min=0,
        max=2)
    travel_time: fields.Integer = fields.Integer(
        title="Travel time",
        description=
        "Home to school travel time (0 - <15 min, 1 - 15~30 min, 2 - 30~60 min, 3 - >60 min)",
        min=0,
        max=3)
    study_time: fields.Integer = fields.Integer(
        title="Study time",
        description=
        "Weekly study time (0 - <2 hrs, 1 - 2~5 hrs, 2 - 5~10 hrs, 3 - >10 hrs)",
        min=0,
        max=3)
    failures: fields.Integer = fields.Integer(
        title="Failures",
        description="Number of past class failures",
        min=1,
        max=4)
    school_support: fields.Boolean = fields.Boolean(
        title="School support",
        description="Extra educational support (true - yes, false - no)")
    family_support: fields.Boolean = fields.Boolean(
        title="Family support",
        description="Family educational support (true - yes, false - no)")
    paid: fields.Boolean = fields.Boolean(
        title="Paid",
        description=
        "Extra paid classes within the course subject (true - yes, false - no)"
    )
    activities: fields.Boolean = fields.Boolean(
        title="Activities",
        description=
        "Participated in extra-curricular activities (true - yes, false - no)")
    nursery: fields.Boolean = fields.Boolean(
        title="Nursery",
        description="Attended nursery school (true - yes, false - no)")
    higher: fields.Boolean = fields.Boolean(
        title="Higher",
        description="Wants to take higher education (true - yes, false - no)")
    Internet: fields.Boolean = fields.Boolean(
        title="Internet",
        description="Internet access at home (true - yes, false - no)")
    Romantic: fields.Boolean = fields.Boolean(
        title="Romantic",
        description="With a romantic relationship (true - yes, false - no)")
    family_rel: fields.Integer = fields.Integer(
        title="Family relationship",
        description="Quality of family relationship",
        min=1,
        max=5)
    free_time: fields.Integer = fields.Integer(
        title="Free time", description="Free time after school", min=1, max=5)
    going_out: fields.Integer = fields.Integer(
        title="Going out amount",
        description="Going out with friends",
        min=1,
        max=5)
    workday_alcohol: fields.Integer = fields.Integer(
        title="Alcohol (workday)",
        description="Workday alcohol consumption",
        min=1,
        max=5)
    weekend_alcohol: fields.Integer = fields.Integer(
        title="Alcohol (weekend)",
        description="Weekend alcohol consumption",
        min=1,
        max=5)
    health: fields.Integer = fields.Integer(
        title="Health", description="Current health status", min=1, max=5)
    absences: fields.Integer = fields.Integer(
        title="Absences",
        description="Number of school absences",
        min=0,
        max=93)
