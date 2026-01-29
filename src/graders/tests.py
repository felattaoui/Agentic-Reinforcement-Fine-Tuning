"""
Test suite for the planner grader.

Run these tests before training to ensure the grader behaves as expected.

The grader uses F2 score with a 3% penalty per extra tool.
Formula: F2 = 5 * (precision * recall) / (4 * precision + recall)
Penalty: -3% per extra tool predicted
Final Score: max(0, F2 - penalty)

F2 prioritizes recall over precision, appropriate for retail customer service
where missing a tool causes workflow failure. The penalty discourages over-prediction.
"""

import json
from src.graders.grader import grade


def test_grader(verbose: bool = True) -> bool:
    """
    Run all grader tests and return success status.

    Args:
        verbose: If True, print detailed results

    Returns:
        bool: True if all tests pass
    """
    all_passed = True

    # Reference item for most tests
    reference_item = {
        "reference_answer": json.dumps({
            "expected_tools": [
                "find_user_id_by_name_zip",
                "get_order_details",
                "get_product_details",
                "exchange_delivered_order_items"
            ]
        })
    }

    # NOTE: With F2 score grader (β=2) + 3% penalty per extra tool:
    # F2 = 5 * (precision * recall) / (4 * precision + recall)
    # Final = max(0, F2 - 0.03 * extra_tools)

    tests = [
        # Test 1: Perfect response (all tools)
        # Recall = 4/4 = 1.0, Precision = 4/4 = 1.0
        # F2 = 5 * (1.0 * 1.0) / (4 * 1.0 + 1.0) = 5/5 = 1.0
        {
            "name": "Perfect response",
            "sample": {"output_text": """
                I'll help with this exchange. Here's my plan:
                1. find_user_id_by_name_zip - to identify the customer
                2. get_order_details - to see the order
                3. get_product_details - to check available options
                4. exchange_delivered_order_items - to process the exchange
            """},
            "item": reference_item,
            "expected": 1.0,
            "tolerance": 0.01
        },

        # Test 2: Missing one tool (3/4)
        # Recall = 3/4 = 0.75, Precision = 3/3 = 1.0
        # F2 = 5 * (1.0 * 0.75) / (4 * 1.0 + 0.75) = 3.75 / 4.75 = 0.789
        {
            "name": "Missing 1 tool (3/4)",
            "sample": {"output_text": """
                Plan:
                1. find_user_id_by_name_zip
                2. get_order_details
                3. exchange_delivered_order_items
            """},  # Missing get_product_details
            "item": reference_item,
            "expected": 0.789,
            "tolerance": 0.01
        },

        # Test 3: All tools present (order doesn't matter)
        # Recall = 4/4 = 1.0, Precision = 4/4 = 1.0
        # F2 = 5 * (1.0 * 1.0) / (4 * 1.0 + 1.0) = 1.0
        {
            "name": "All tools (any order)",
            "sample": {"output_text": """
                Plan:
                1. exchange_delivered_order_items - do the exchange first
                2. find_user_id_by_name_zip - then identify user
                3. get_order_details
                4. get_product_details
            """},
            "item": reference_item,
            "expected": 1.0,
            "tolerance": 0.01
        },

        # Test 4: One extra tool
        # Recall = 4/4 = 1.0, Precision = 4/5 = 0.8
        # F2 = 5 * (0.8 * 1.0) / (4 * 0.8 + 1.0) = 4.0 / 4.2 = 0.952
        # Penalty = 1 * 0.03 = 0.03
        # Final = 0.952 - 0.03 = 0.922
        {
            "name": "One extra tool",
            "sample": {"output_text": """
                Plan:
                1. find_user_id_by_name_zip
                2. get_user_details
                3. get_order_details
                4. get_product_details
                5. exchange_delivered_order_items
            """},
            "item": reference_item,
            "expected": 0.922,
            "tolerance": 0.01
        },

        # Test 5: Empty response
        {
            "name": "Empty response",
            "sample": {"output_text": ""},
            "item": reference_item,
            "expected": 0.0,
            "tolerance": 0.01
        },

        # Test 6: Completely wrong tools
        # Recall = 0/4 = 0, Precision = 0/2 = 0
        # F1 = 0 (no matching tools)
        {
            "name": "Wrong tools",
            "sample": {"output_text": "I'll use calculate and transfer_to_human_agents"},
            "item": reference_item,
            "expected": 0.0,
            "tolerance": 0.01
        },

        # Test 7: Two extra tools
        # Recall = 4/4 = 1.0, Precision = 4/6 = 0.667
        # F2 = 5 * (0.667 * 1.0) / (4 * 0.667 + 1.0) = 3.333 / 3.667 = 0.909
        # Penalty = 2 * 0.03 = 0.06
        # Final = 0.909 - 0.06 = 0.849
        {
            "name": "Two extra tools",
            "sample": {"output_text": """
                1. find_user_id_by_name_zip
                2. get_user_details
                3. get_order_details
                4. list_all_product_types
                5. get_product_details
                6. exchange_delivered_order_items
            """},
            "item": reference_item,
            "expected": 0.849,
            "tolerance": 0.02
        },

        # Test 8: No expected tools, no prediction (edge case)
        {
            "name": "No expected tools, no prediction",
            "sample": {"output_text": "No tools needed"},
            "item": {"reference_answer": json.dumps({"expected_tools": []})},
            "expected": 1.0,
            "tolerance": 0.01
        },

        # Test 9: No expected tools but prediction made
        {
            "name": "No expected tools, prediction made",
            "sample": {"output_text": "I'll use get_user_details"},
            "item": {"reference_answer": json.dumps({"expected_tools": []})},
            "expected": 0.0,  # Penalty for predicting when nothing expected
            "tolerance": 0.01
        },

        # Test 10: Half tools correct
        # Recall = 2/4 = 0.5, Precision = 2/2 = 1.0
        # F2 = 5 * (1.0 * 0.5) / (4 * 1.0 + 0.5) = 2.5 / 4.5 = 0.556
        {
            "name": "Half tools correct",
            "sample": {"output_text": """
                1. find_user_id_by_name_zip
                2. exchange_delivered_order_items
            """},
            "item": reference_item,
            "expected": 0.556,
            "tolerance": 0.01
        },
    ]

    # Run tests
    for i, test in enumerate(tests, 1):
        score = grade(test["sample"], test["item"])
        passed = abs(score - test["expected"]) <= test["tolerance"]

        if not passed:
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"Test {i} - {test['name']}: {score:.3f} (expected {test['expected']}) [{status}]")

    if verbose:
        print()
        if all_passed:
            print("All grader tests passed!")
        else:
            print("Some tests failed!")

    return all_passed


def test_grader_json_output(verbose: bool = True) -> bool:
    """
    Test grader with JSON structured output format.

    Args:
        verbose: If True, print detailed results

    Returns:
        bool: True if all tests pass
    """
    all_passed = True

    reference_item = {
        "reference_answer": json.dumps({
            "expected_tools": [
                "find_user_id_by_name_zip",
                "get_order_details",
                "get_product_details",
                "exchange_delivered_order_items"
            ]
        })
    }

    json_tests = [
        # Test 1: Perfect JSON response
        {
            "name": "JSON - Perfect response",
            "sample": {"output_text": json.dumps({
                "reasoning": "I need to find the user, get order details, check products, and process exchange",
                "tools": [
                    "find_user_id_by_name_zip",
                    "get_order_details",
                    "get_product_details",
                    "exchange_delivered_order_items"
                ]
            })},
            "item": reference_item,
            "expected": 1.0,
            "tolerance": 0.01
        },

        # Test 2: JSON with missing tool
        # Recall = 3/4 = 0.75, Precision = 3/3 = 1.0
        # F2 = 5 * (1.0 * 0.75) / (4 * 1.0 + 0.75) = 3.75 / 4.75 = 0.789
        {
            "name": "JSON - Missing 1 tool",
            "sample": {"output_text": json.dumps({
                "reasoning": "Processing exchange",
                "tools": [
                    "find_user_id_by_name_zip",
                    "get_order_details",
                    "exchange_delivered_order_items"
                ]
            })},
            "item": reference_item,
            "expected": 0.789,
            "tolerance": 0.01
        },

        # Test 3: JSON with extra tool
        # Recall = 4/4 = 1.0, Precision = 4/5 = 0.8
        # F2 = 5 * (0.8 * 1.0) / (4 * 0.8 + 1.0) = 4.0 / 4.2 = 0.952
        # Penalty = 1 * 0.03 = 0.03
        # Final = 0.952 - 0.03 = 0.922
        {
            "name": "JSON - One extra tool",
            "sample": {"output_text": json.dumps({
                "reasoning": "Complete plan",
                "tools": [
                    "find_user_id_by_name_zip",
                    "get_user_details",  # Extra
                    "get_order_details",
                    "get_product_details",
                    "exchange_delivered_order_items"
                ]
            })},
            "item": reference_item,
            "expected": 0.922,
            "tolerance": 0.01
        },

        # Test 4: JSON with empty tools array
        {
            "name": "JSON - Empty tools array",
            "sample": {"output_text": json.dumps({
                "reasoning": "No tools needed",
                "tools": []
            })},
            "item": reference_item,
            "expected": 0.0,
            "tolerance": 0.01
        },

        # Test 5: Mixed case tool names in JSON
        {
            "name": "JSON - Mixed case tools",
            "sample": {"output_text": json.dumps({
                "reasoning": "Test",
                "tools": [
                    "FIND_USER_ID_BY_NAME_ZIP",
                    "Get_Order_Details",
                    "get_product_details",
                    "exchange_delivered_order_items"
                ]
            })},
            "item": reference_item,
            "expected": 1.0,
            "tolerance": 0.01
        },
    ]

    for i, test in enumerate(json_tests, 1):
        score = grade(test["sample"], test["item"])
        passed = abs(score - test["expected"]) <= test["tolerance"]

        if not passed:
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"JSON Test {i} - {test['name']}: {score:.3f} (expected {test['expected']}) [{status}]")

    if verbose:
        print()
        if all_passed:
            print("All JSON grader tests passed!")
        else:
            print("Some JSON tests failed!")

    return all_passed


def test_grader_edge_cases(verbose: bool = True) -> bool:
    """
    Test edge cases that might cause crashes during training.

    Args:
        verbose: If True, print detailed results

    Returns:
        bool: True if all tests pass
    """
    all_passed = True

    edge_cases = [
        # Invalid JSON in reference
        {
            "name": "Invalid reference JSON",
            "sample": {"output_text": "get_order_details"},
            "item": {"reference_answer": "not valid json"},
            "should_not_crash": True
        },

        # Missing output_text
        {
            "name": "Missing output_text",
            "sample": {},
            "item": {"reference_answer": json.dumps({"expected_tools": ["get_order_details"]})},
            "should_not_crash": True
        },

        # Missing reference_answer
        {
            "name": "Missing reference_answer",
            "sample": {"output_text": "get_order_details"},
            "item": {},
            "should_not_crash": True
        },

        # None values
        {
            "name": "None output_text",
            "sample": {"output_text": None},
            "item": {"reference_answer": json.dumps({"expected_tools": []})},
            "should_not_crash": True
        },

        # Unicode and special characters
        {
            "name": "Unicode in response",
            "sample": {"output_text": "get_order_details with special chars"},
            "item": {"reference_answer": json.dumps({"expected_tools": ["get_order_details"]})},
            "should_not_crash": True
        },

        # Malformed JSON output (should fallback to text parsing)
        {
            "name": "Malformed JSON output (fallback to text)",
            "sample": {"output_text": '{"tools": ["get_order_details"'},  # Missing closing
            "item": {"reference_answer": json.dumps({"expected_tools": ["get_order_details"]})},
            "should_not_crash": True
        },

        # JSON without tools key (should fallback to text parsing)
        {
            "name": "JSON without tools key",
            "sample": {"output_text": '{"reasoning": "test", "actions": ["get_order_details"]}'},
            "item": {"reference_answer": json.dumps({"expected_tools": ["get_order_details"]})},
            "should_not_crash": True
        },
    ]

    for i, test in enumerate(edge_cases, 1):
        try:
            score = grade(test["sample"], test["item"])
            passed = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
        except Exception as e:
            passed = False
            if verbose:
                print(f"  Exception: {e}")

        if not passed:
            all_passed = False

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"Edge case {i} - {test['name']}: [{status}]")

    if verbose:
        print()
        if all_passed:
            print("All edge case tests passed!")
        else:
            print("Some edge case tests failed!")

    return all_passed


def run_all_tests(verbose: bool = True) -> bool:
    """
    Run all grader tests.

    Args:
        verbose: If True, print detailed results

    Returns:
        bool: True if all tests pass
    """
    if verbose:
        print("=" * 60)
        print("GRADER TESTS (F2 Score) - Text Parsing")
        print("=" * 60)
        print()

    basic_passed = test_grader(verbose)

    if verbose:
        print()
        print("-" * 60)
        print("GRADER TESTS (F2 Score) - JSON Structured Output")
        print("-" * 60)
        print()

    json_passed = test_grader_json_output(verbose)

    if verbose:
        print()
        print("-" * 60)
        print("EDGE CASE TESTS")
        print("-" * 60)
        print()

    edge_passed = test_grader_edge_cases(verbose)

    return basic_passed and json_passed and edge_passed


if __name__ == "__main__":
    run_all_tests()
