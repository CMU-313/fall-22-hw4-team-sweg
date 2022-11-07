from dataclasses import dataclass

from flask_restx import fields


@dataclass(frozen=True)
class Applicant:
    school: str
    sex: str
    age: int
    address: str
    family_size: str
    p_status: str
    mother_edu: int
    father_edu: int
    mother_job: str
    father_job: str
    reason: str
    guardian: str
    travel_time: int
    study_time: int
    failures: int
    school_support: str
    family_support: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    family_rel: int
    free_time: int
    going_out: int
    workday_alcohol: int
    weekend_alcohol: int
    health: int
    absences: int


@dataclass(frozen=True)
class ApplicantFields:
    school: fields.Boolean = fields.String(
        title="School",
        description="Student's school (GP - Gabriel Pereira, MS - Mousinho da Silveira)",
        enum=["GP", "MS"],
        required=True,
    )
    sex: fields.Boolean = fields.String(
        title="Sex",
        description="Student's sex (F - female, M - male)",
        enum=["F", "M"],
        required=True,
    )
    age: fields.Integer = fields.Integer(
        title="Age", description="Student's age", min=15, max=22, required=True
    )
    address: fields.Boolean = fields.String(
        title="Address",
        description="Student's home address type (U - urban, R - rural)",
        enum=["U", "R"],
        required=True,
    )
    family_size: fields.Boolean = fields.String(
        title="Family size",
        description="Family size (LE3 - <=3, GT3 - >3)",
        enum=["LE3", "GT3"],
        required=True,
    )
    p_status: fields.Boolean = fields.String(
        title="P status",
        description="Parent's cohabitation status (T - living together, A - apart)",
        enum=["T", "A"],
        required=True,
    )
    mother_edu: fields.Integer = fields.Integer(
        title="Mother's education",
        description="Mother's education level (0 - none, 1 - 4th grade, 2 - 5~9th grade, 3 - secondary, 4 - higher)",
        min=0,
        max=4,
        required=True,
    )
    father_edu: fields.Integer = fields.Integer(
        title="Father's education",
        description="Father's education level (0 - none, 1 - 4th grade, 2 - 5~9th grade, 3 - secondary, 4 - higher)",
        min=0,
        max=4,
        required=True,
    )
    mother_job: fields.Integer = fields.String(
        title="Mother's job",
        description="Mother's job (teacher, health care related, civil services (e.g. administrative or police), at_home or other)",
        enum=["teacher", "health", "services", "at home", "other"],
        required=True,
    )
    father_job: fields.Integer = fields.String(
        title="Father's job",
        description="Father's job (teacher, health care related, civil services (e.g. administrative or police), at_home or other)",
        enum=["teacher", "health", "services", "at home", "other"],
        required=True,
    )
    reason: fields.Integer = fields.String(
        title="Reason",
        description="Reason to choose this school (close to home, school reputation, course preference or other)",
        enum=["home", "reputation", "course", "other"],
        required=True,
    )
    guardian: fields.Integer = fields.String(
        title="Guardian",
        description="Student's guardian (mother, father, or other)",
        enum=["mother", "father", "other"],
        required=True,
    )
    travel_time: fields.Integer = fields.Integer(
        title="Travel time",
        description="Home to school travel time (1 - <15 min, 2 - 15~30 min, 3 - 30~60 min, 4 - >60 min)",
        min=1,
        max=4,
        required=True,
    )
    study_time: fields.Integer = fields.Integer(
        title="Study time",
        description="Weekly study time (1 - <2 hrs, 2 - 2~5 hrs, 3 - 5~10 hrs, 4 - >10 hrs)",
        min=1,
        max=4,
        required=True,
    )
    failures: fields.Integer = fields.Integer(
        title="Failures",
        description="Number of past class failures",
        min=1,
        max=4,
        required=True,
    )
    school_support: fields.Boolean = fields.String(
        title="School support",
        description="Extra educational support (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    family_support: fields.Boolean = fields.String(
        title="Family support",
        description="Family educational support (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    paid: fields.Boolean = fields.String(
        title="Paid",
        description="Extra paid classes within the course subject (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    activities: fields.Boolean = fields.String(
        title="Activities",
        description="Participated in extra-curricular activities (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    nursery: fields.Boolean = fields.String(
        title="Nursery",
        description="Attended nursery school (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    higher: fields.Boolean = fields.String(
        title="Higher",
        description="Wants to take higher education (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    internet: fields.Boolean = fields.String(
        title="Internet",
        description="Internet access at home (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    romantic: fields.Boolean = fields.String(
        title="Romantic",
        description="With a romantic relationship (yes or no)",
        enum=["yes", "no"],
        required=True,
    )
    family_rel: fields.Integer = fields.Integer(
        title="Family relationship",
        description="Quality of family relationship",
        min=1,
        max=5,
        required=True,
    )
    free_time: fields.Integer = fields.Integer(
        title="Free time",
        description="Free time after school",
        min=1,
        max=5,
        required=True,
    )
    going_out: fields.Integer = fields.Integer(
        title="Going out amount",
        description="Going out with friends",
        min=1,
        max=5,
        required=True,
    )
    workday_alcohol: fields.Integer = fields.Integer(
        title="Alcohol (workday)",
        description="Workday alcohol consumption",
        min=1,
        max=5,
        required=True,
    )
    weekend_alcohol: fields.Integer = fields.Integer(
        title="Alcohol (weekend)",
        description="Weekend alcohol consumption",
        min=1,
        max=5,
        required=True,
    )
    health: fields.Integer = fields.Integer(
        title="Health", description="Current health status", min=1, max=5, required=True
    )
    absences: fields.Integer = fields.Integer(
        title="Absences",
        description="Number of school absences",
        min=0,
        max=93,
        required=True,
    )
