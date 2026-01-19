from HouseholdDecisionRule import HouseholdDecisionRule



"""
    A HouseholdDecisionRule that applies noise to the results of
    an existing HouseholdDecisionRule.

    This algorithm operates as follows:
    (a) Noise, N, is drawn from a Gaussian distribution with mean zero and
        custom standard deviation (interpreted from code usage).
    (b) N is clamped in the range [min, max] for customizable values min, max.
    (c) Another HouseholdDecisionRule is called, and returns value D.
    (d) This HouseholdDecisionRule returns D * (1 + N).
    """
class NoisyHouseholdDecisionRule(HouseholdDecisionRule):
    
    # Note: Python doesn't use Java's Guice annotations like @AssistedInject, @Assisted, @Named,
    # @BindingAnnotation, or @Retention. The standard Python constructor (__init__)
    # handles dependency injection and parameter passing directly.

    def __init__(self,
                 rule: HouseholdDecisionRule,
                 noise_aggression: float, # Corresponds to noiseAggressiveness
                 noise_lower_bound: float, # Corresponds to noiseLowerBound
                 noise_upper_bound: float # Corresponds to noiseUpperBound
                ):
        """
        Create a NoisyHouseholdDecisionRule.

        Args:
            rule: The HouseholdDecisionRule to which noise is added.
            noise_aggression: The standard deviation of the Gaussian distribution
                              from which to draw noise. (The Java comment says "variance",
                               but the code `random.nextGaussian() * noiseAggressiveness`
                              implies noiseAggressiveness is used as the standard deviation).
                              Must be non-negative.
            noise_lower_bound: A lower bound on the value of the noise factor (N) to apply.
            noise_upper_bound: An upper bound on the value of the noise factor (N) to apply.

        Raises:
            ValueError: If input arguments are invalid (e.g., rule is None,
                        noise_aggression is negative, or bounds are inverted).
        """
        # Call the parent class constructor
        super().__init__()

        # Corresponds to StateVerifier.checkNotNull(rule)
        check_not_null(rule, "The base HouseholdDecisionRule cannot be None")

        # Corresponds to Preconditions.checkArgument(noiseAggressiveness >= 0.)
        check_argument(noise_aggression >= 0., "Noise aggressiveness (std dev) must be non-negative")

        # Corresponds to Preconditions.checkArgument(noiseLowerBound <= noiseUpperBound)
        check_argument(noise_lower_bound <= noise_upper_bound,
                       "Noise lower bound cannot be greater than noise upper bound")

        # Store instance variables (final in Java means effectively read-only after init)
        self.rule: HouseholdDecisionRule = rule
        self.noise_aggression: float = noise_aggression
        self.noise_lower_bound: float = noise_lower_bound
        self.noise_upper_bound: float = noise_upper_bound

        # Note on noise_aggression: The Java code multiplies random.nextGaussian()
        # (which returns a sample from a standard normal distribution, mean 0, std dev 1)
        # by noiseAggressiveness. This means noiseAggressiveness is used as the standard deviation
        # for the resulting noise, despite the Java comment calling it "variance".
        # If it *were* strictly variance V, the multiplication should be by sqrt(V).
        # We follow the code's apparent logic, treating noise_aggression as standard deviation.


    def compute_next(self, current_state: HouseholdState) -> float:
        """
        Computes the next value by applying noise to the base rule's result.

        Args:
            current_state: The current state of the household.

        Returns:
            The computed decision value with applied noise.
        """
        # (c) Another HouseholdDecisionRule is called, and returns value D;
        base_value: float = self.rule.compute_next(current_state)

        # Get the random number generator from the simulation model (mimicking Java's static access)
        sim_random = get_running_simulation_model().random

        # (a) Noise, N, is drawn from a Gaussian distribution with mean zero and custom standard deviation;
        # The noise_aggression is used as the standard deviation here.
        noise: float = sim_random.gauss(0., self.noise_aggression)

        # (b) N is clamped in the range [min, max];
        clamped_noise: float = clamp(self.noise_lower_bound, noise, self.noise_upper_bound)

        # (d) This HouseholdDecisionRule returns D * (1 + N).
        # Apply the noise factor (1 + N) to the base value.
        result: float = base_value * (1.0 + clamped_noise)

        # Call the parent's method to record the value (corresponding to super.recordNewValue)
        super().record_new_value(result)

        return result

    def __str__(self) -> str:
        """
        Returns a brief description of this object.
        Corresponds to the Java toString() method.
        """
        return (f"Noisy Household Decision Rule, applies to: {self.rule}"
                f", noise aggressiveness (std dev): {self.noise_aggression}.")

    def __repr__(self) -> str:
        """
        Provides a more detailed representation, useful for debugging.
        """
        return (f"NoisyHouseholdDecisionRule(rule={self.rule!r}, "
                f"noise_aggression={self.noise_aggression}, "
                f"noise_lower_bound={self.noise_lower_bound}, "
                f"noise_upper_bound={self.noise_upper_bound})")




# # --- Example Usage ---
# if __name__ == "__main__":
#     # Define a simple base rule for demonstration
#     class SimpleBaseRule(HouseholdDecisionRule):
#         """A simple rule that always returns a fixed value."""
#         def __init__(self, fixed_value: float):
#             super().__init__()
#             self._fixed_value = fixed_value

#         def compute_next(self, current_state: HouseholdState) -> float:
#             """Returns the fixed value regardless of state."""
#             # In a real rule, this would use the current_state
#             return self._fixed_value

#         def __str__(self) -> str:
#             return f"Simple Base Rule ({self._fixed_value})"

#         def __repr__(self) -> str:
#              return f"SimpleBaseRule({self._fixed_value})"


#     # Create an instance of the simple base rule that returns 10.0
#     base_rule_instance = SimpleBaseRule(10.0)

#     # Create an instance of the noisy rule
#     # Let's set parameters: std dev 0.1, noise clamped between -0.2 and 0.2
#     noisy_rule = NoisyHouseholdDecisionRule(
#         rule=base_rule_instance,
#         noise_aggression=0.1, # Standard deviation of Gaussian noise
#         noise_lower_bound=-0.2, # Noise factor N will be >= -0.2
#         noise_upper_bound=0.2   # Noise factor N will be <= 0.2
#     )

#     # Create a dummy household state (not used by SimpleBaseRule, but required by compute_next signature)
#     dummy_state = HouseholdState()

#     print(noisy_rule) # Print the string representation using __str__
#     print(repr(noisy_rule)) # Print the representation using __repr__

#     # Compute the next value several times to see the noise effect
#     print("\nComputing values with noise (std dev 0.1, clamp [-0.2, 0.2]):")
#     for i in range(5):
#         computed_value = noisy_rule.compute_next(dummy_state)
#         print(f"Step {i+1}: Computed value = {computed_value:.4f}") # Format for readability

#     # Demonstrate clamping effects with higher noise aggression and tighter bounds
#     print("\nComputing values with higher noise (std dev 0.5, clamp [-0.1, 0.1]):")
#     noisy_rule_clamped = NoisyHouseholdDecisionRule(
#         rule=base_rule_instance,
#         noise_aggression=0.5, # Higher std dev
#         noise_lower_bound=-0.1, # Tighter lower bound for N
#         noise_upper_bound=0.1   # Tighter upper bound for N
#     )

#     print(noisy_rule_clamped)
#     for i in range(5):
#         computed_value = noisy_rule_clamped.compute_next(dummy_state)
#         print(f"Step {i+1}: Computed value = {computed_value:.4f}") # Format for readability

#     # Demonstrate error handling for invalid constructor arguments
#     print("\nDemonstrating error handling:")
#     try:
#         # Invalid: rule is None
#         print("Attempting to create rule with None rule...")
#         NoisyHouseholdDecisionRule(None, 0.1, -0.1, 0.1)
#     except ValueError as e:
#         print(f"Caught expected error: {e}")

#     try:
#         # Invalid: noise_aggression is negative
#         print("Attempting to create rule with negative noise aggression...")
#         NoisyHouseholdDecisionRule(base_rule_instance, -0.1, -0.1, 0.1)
#     except ValueError as e:
#         print(f"Caught expected error: {e}")

#     try:
#         # Invalid: lower_bound > upper_bound
#         print("Attempting to create rule with inverted bounds...")
#         NoisyHouseholdDecisionRule(base_rule_instance, 0.1, 0.1, -0.1)
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
