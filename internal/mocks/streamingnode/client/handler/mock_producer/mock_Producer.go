// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_producer

import (
	context "context"

	message "github.com/milvus-io/milvus/pkg/streaming/util/message"
	mock "github.com/stretchr/testify/mock"

	types "github.com/milvus-io/milvus/pkg/streaming/util/types"
)

// MockProducer is an autogenerated mock type for the Producer type
type MockProducer struct {
	mock.Mock
}

type MockProducer_Expecter struct {
	mock *mock.Mock
}

func (_m *MockProducer) EXPECT() *MockProducer_Expecter {
	return &MockProducer_Expecter{mock: &_m.Mock}
}

// Assignment provides a mock function with given fields:
func (_m *MockProducer) Assignment() types.PChannelInfoAssigned {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Assignment")
	}

	var r0 types.PChannelInfoAssigned
	if rf, ok := ret.Get(0).(func() types.PChannelInfoAssigned); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(types.PChannelInfoAssigned)
	}

	return r0
}

// MockProducer_Assignment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Assignment'
type MockProducer_Assignment_Call struct {
	*mock.Call
}

// Assignment is a helper method to define mock.On call
func (_e *MockProducer_Expecter) Assignment() *MockProducer_Assignment_Call {
	return &MockProducer_Assignment_Call{Call: _e.mock.On("Assignment")}
}

func (_c *MockProducer_Assignment_Call) Run(run func()) *MockProducer_Assignment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProducer_Assignment_Call) Return(_a0 types.PChannelInfoAssigned) *MockProducer_Assignment_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProducer_Assignment_Call) RunAndReturn(run func() types.PChannelInfoAssigned) *MockProducer_Assignment_Call {
	_c.Call.Return(run)
	return _c
}

// Available provides a mock function with given fields:
func (_m *MockProducer) Available() <-chan struct{} {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Available")
	}

	var r0 <-chan struct{}
	if rf, ok := ret.Get(0).(func() <-chan struct{}); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(<-chan struct{})
		}
	}

	return r0
}

// MockProducer_Available_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Available'
type MockProducer_Available_Call struct {
	*mock.Call
}

// Available is a helper method to define mock.On call
func (_e *MockProducer_Expecter) Available() *MockProducer_Available_Call {
	return &MockProducer_Available_Call{Call: _e.mock.On("Available")}
}

func (_c *MockProducer_Available_Call) Run(run func()) *MockProducer_Available_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProducer_Available_Call) Return(_a0 <-chan struct{}) *MockProducer_Available_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProducer_Available_Call) RunAndReturn(run func() <-chan struct{}) *MockProducer_Available_Call {
	_c.Call.Return(run)
	return _c
}

// Close provides a mock function with given fields:
func (_m *MockProducer) Close() {
	_m.Called()
}

// MockProducer_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockProducer_Close_Call struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
func (_e *MockProducer_Expecter) Close() *MockProducer_Close_Call {
	return &MockProducer_Close_Call{Call: _e.mock.On("Close")}
}

func (_c *MockProducer_Close_Call) Run(run func()) *MockProducer_Close_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProducer_Close_Call) Return() *MockProducer_Close_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockProducer_Close_Call) RunAndReturn(run func()) *MockProducer_Close_Call {
	_c.Call.Return(run)
	return _c
}

// IsAvailable provides a mock function with given fields:
func (_m *MockProducer) IsAvailable() bool {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for IsAvailable")
	}

	var r0 bool
	if rf, ok := ret.Get(0).(func() bool); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(bool)
	}

	return r0
}

// MockProducer_IsAvailable_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'IsAvailable'
type MockProducer_IsAvailable_Call struct {
	*mock.Call
}

// IsAvailable is a helper method to define mock.On call
func (_e *MockProducer_Expecter) IsAvailable() *MockProducer_IsAvailable_Call {
	return &MockProducer_IsAvailable_Call{Call: _e.mock.On("IsAvailable")}
}

func (_c *MockProducer_IsAvailable_Call) Run(run func()) *MockProducer_IsAvailable_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProducer_IsAvailable_Call) Return(_a0 bool) *MockProducer_IsAvailable_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProducer_IsAvailable_Call) RunAndReturn(run func() bool) *MockProducer_IsAvailable_Call {
	_c.Call.Return(run)
	return _c
}

// Produce provides a mock function with given fields: ctx, msg
func (_m *MockProducer) Produce(ctx context.Context, msg message.MutableMessage) (*types.AppendResult, error) {
	ret := _m.Called(ctx, msg)

	if len(ret) == 0 {
		panic("no return value specified for Produce")
	}

	var r0 *types.AppendResult
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, message.MutableMessage) (*types.AppendResult, error)); ok {
		return rf(ctx, msg)
	}
	if rf, ok := ret.Get(0).(func(context.Context, message.MutableMessage) *types.AppendResult); ok {
		r0 = rf(ctx, msg)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*types.AppendResult)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, message.MutableMessage) error); ok {
		r1 = rf(ctx, msg)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockProducer_Produce_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Produce'
type MockProducer_Produce_Call struct {
	*mock.Call
}

// Produce is a helper method to define mock.On call
//   - ctx context.Context
//   - msg message.MutableMessage
func (_e *MockProducer_Expecter) Produce(ctx interface{}, msg interface{}) *MockProducer_Produce_Call {
	return &MockProducer_Produce_Call{Call: _e.mock.On("Produce", ctx, msg)}
}

func (_c *MockProducer_Produce_Call) Run(run func(ctx context.Context, msg message.MutableMessage)) *MockProducer_Produce_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(message.MutableMessage))
	})
	return _c
}

func (_c *MockProducer_Produce_Call) Return(_a0 *types.AppendResult, _a1 error) *MockProducer_Produce_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockProducer_Produce_Call) RunAndReturn(run func(context.Context, message.MutableMessage) (*types.AppendResult, error)) *MockProducer_Produce_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockProducer creates a new instance of MockProducer. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockProducer(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockProducer {
	mock := &MockProducer{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}