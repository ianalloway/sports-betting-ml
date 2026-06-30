name: Feature request
description: Suggest a new feature, model improvement, or UI enhancement
labels: [enhancement]
body:
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear description of what the problem is.
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
  - type: textarea
    id: additional
    attributes:
      label: Additional context
