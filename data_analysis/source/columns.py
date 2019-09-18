
class Columns(object):

    def __init__(self):
        self.event_id = 'Event.Id'
        self.investigation_type = 'Investigation.Type'
        self.accident_number = 'Accident.Number'
        self.event_date = 'Event.Date'
        self.location = 'Location'
        self.country = 'Country'
        self.latitude = "Latitude"
        self.longitude = 'Longitude'
        self.airport_code = 'Airport.Code'
        self.airport_name = 'Airport.Name'
        self.injury_severity = 'Injury.Severity'
        self.aircraft_damage = 'Aircraft.Damage'
        self.aircraft_category = 'Aircraft.Category'
        self.registration_no = 'Registration.Number'
        self.make = 'Make'
        self.model = 'Model'
        self.amateur_built = 'Amateur.Built'
        self.number_of_engines = 'Number.of.Engines'
        self.engine_type = 'Engine.Type'
        self.far_description = 'FAR.Description'
        self.schedule = "Schedule"
        self.purpose_of_flight = 'Purpose.of.Flight'
        self.air_carrier = 'Air.Carrier'
        self.total_fatal_injuries = 'Total.Fatal.Injuries'
        self.total_serious_injuries = 'Total.Serious.Injuries'
        self.total_minor_injuries = 'Total.Minor.Injuries'
        self.total_uninjured = 'Total.Uninjured'
        self.weather_condition = 'Weather.Condition'
        self.broad_phase_of_flight = 'Broad.Phase.of.Flight'
        self.report_status = 'Report.Status'
        self.publication_date = 'Publication.Date'

    def encode_features(self):
        features = [self.make, self.model, self.amateur_built, self.number_of_engines, self.engine_type, self.air_carrier, self.aircraft_damage,
                    self.total_fatal_injuries, self.total_serious_injuries, self.total_minor_injuries, self.total_uninjured,
                    self.weather_condition, self.broad_phase_of_flight]
        return features
