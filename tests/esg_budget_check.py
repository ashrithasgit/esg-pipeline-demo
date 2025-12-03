# tests/esg_budget_check.py
import argparse, json, time
from codecarbon import EmissionsTracker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-kg', type=float, required=True)
    parser.add_argument('--out', type=str, default='reports/esg_report.json')
    args = parser.parse_args()

    tracker = EmissionsTracker(measure_power_secs=1, project_name='esg-test')
    tracker.start()
    time.sleep(2)
    emissions = tracker.stop()
    report = {'runtime_sec': 2, 'emissions_kg': emissions}
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print('ESG report:', report)
    if emissions > args.max_kg:
        sys.exit(3)

if __name__ == '__main__':
    main()
