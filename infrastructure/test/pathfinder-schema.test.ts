import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { BUNDLING_STACKS } from 'aws-cdk-lib/cx-api';
import { CdkStack } from '../lib/cdk-stack';
import schema from '../../lambda/pathfinder-tool/tool-schema.json';

// Synthesize locally without Docker/pip bundling or AWS lookups.

test('Gateway advertises the packaged score-aware contract and real Lambda source', () => {
  const app = new cdk.App({ context: { [BUNDLING_STACKS]: [] } });
  const stack = new CdkStack(app, 'PathfinderIntegrationTest');
  const template = Template.fromStack(stack);
  const targets = template.findResources('AWS::BedrockAgentCore::GatewayTarget');
  const target = Object.values(targets).find(value => value.Properties.Name === 'AgentCoreGatewayTool-Pathfinder');
  expect(target?.Properties.TargetConfiguration.Mcp.Lambda.ToolSchema.InlinePayload).toEqual([schema]);
  expect(schema.InputSchema.Properties.strategy.Description).toContain('score_aware');
  expect(schema.InputSchema.Properties.planning.Type).toBe('object');
  template.hasResourceProperties('AWS::Lambda::Function', {
    FunctionName: 'AgentCoreGatewayTool-Pathfinder', Handler: 'index.lambda_handler', Runtime: 'python3.12',
  });
});
