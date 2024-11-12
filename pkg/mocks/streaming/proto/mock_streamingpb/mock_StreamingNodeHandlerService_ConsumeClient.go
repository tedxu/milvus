// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_streamingpb

import (
	context "context"

	mock "github.com/stretchr/testify/mock"
	metadata "google.golang.org/grpc/metadata"

	streamingpb "github.com/milvus-io/milvus/pkg/streaming/proto/streamingpb"
)

// MockStreamingNodeHandlerService_ConsumeClient is an autogenerated mock type for the StreamingNodeHandlerService_ConsumeClient type
type MockStreamingNodeHandlerService_ConsumeClient struct {
	mock.Mock
}

type MockStreamingNodeHandlerService_ConsumeClient_Expecter struct {
	mock *mock.Mock
}

func (_m *MockStreamingNodeHandlerService_ConsumeClient) EXPECT() *MockStreamingNodeHandlerService_ConsumeClient_Expecter {
	return &MockStreamingNodeHandlerService_ConsumeClient_Expecter{mock: &_m.Mock}
}

// CloseSend provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeClient) CloseSend() error {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for CloseSend")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'CloseSend'
type MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call struct {
	*mock.Call
}

// CloseSend is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) CloseSend() *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call{Call: _e.mock.On("CloseSend")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call) RunAndReturn(run func() error) *MockStreamingNodeHandlerService_ConsumeClient_CloseSend_Call {
	_c.Call.Return(run)
	return _c
}

// Context provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeClient) Context() context.Context {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Context")
	}

	var r0 context.Context
	if rf, ok := ret.Get(0).(func() context.Context); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(context.Context)
		}
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_Context_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Context'
type MockStreamingNodeHandlerService_ConsumeClient_Context_Call struct {
	*mock.Call
}

// Context is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) Context() *MockStreamingNodeHandlerService_ConsumeClient_Context_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_Context_Call{Call: _e.mock.On("Context")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Context_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeClient_Context_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Context_Call) Return(_a0 context.Context) *MockStreamingNodeHandlerService_ConsumeClient_Context_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Context_Call) RunAndReturn(run func() context.Context) *MockStreamingNodeHandlerService_ConsumeClient_Context_Call {
	_c.Call.Return(run)
	return _c
}

// Header provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeClient) Header() (metadata.MD, error) {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Header")
	}

	var r0 metadata.MD
	var r1 error
	if rf, ok := ret.Get(0).(func() (metadata.MD, error)); ok {
		return rf()
	}
	if rf, ok := ret.Get(0).(func() metadata.MD); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(metadata.MD)
		}
	}

	if rf, ok := ret.Get(1).(func() error); ok {
		r1 = rf()
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockStreamingNodeHandlerService_ConsumeClient_Header_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Header'
type MockStreamingNodeHandlerService_ConsumeClient_Header_Call struct {
	*mock.Call
}

// Header is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) Header() *MockStreamingNodeHandlerService_ConsumeClient_Header_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_Header_Call{Call: _e.mock.On("Header")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Header_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeClient_Header_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Header_Call) Return(_a0 metadata.MD, _a1 error) *MockStreamingNodeHandlerService_ConsumeClient_Header_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Header_Call) RunAndReturn(run func() (metadata.MD, error)) *MockStreamingNodeHandlerService_ConsumeClient_Header_Call {
	_c.Call.Return(run)
	return _c
}

// Recv provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeClient) Recv() (*streamingpb.ConsumeResponse, error) {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Recv")
	}

	var r0 *streamingpb.ConsumeResponse
	var r1 error
	if rf, ok := ret.Get(0).(func() (*streamingpb.ConsumeResponse, error)); ok {
		return rf()
	}
	if rf, ok := ret.Get(0).(func() *streamingpb.ConsumeResponse); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*streamingpb.ConsumeResponse)
		}
	}

	if rf, ok := ret.Get(1).(func() error); ok {
		r1 = rf()
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockStreamingNodeHandlerService_ConsumeClient_Recv_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Recv'
type MockStreamingNodeHandlerService_ConsumeClient_Recv_Call struct {
	*mock.Call
}

// Recv is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) Recv() *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_Recv_Call{Call: _e.mock.On("Recv")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call) Return(_a0 *streamingpb.ConsumeResponse, _a1 error) *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call) RunAndReturn(run func() (*streamingpb.ConsumeResponse, error)) *MockStreamingNodeHandlerService_ConsumeClient_Recv_Call {
	_c.Call.Return(run)
	return _c
}

// RecvMsg provides a mock function with given fields: m
func (_m *MockStreamingNodeHandlerService_ConsumeClient) RecvMsg(m interface{}) error {
	ret := _m.Called(m)

	if len(ret) == 0 {
		panic("no return value specified for RecvMsg")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(interface{}) error); ok {
		r0 = rf(m)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RecvMsg'
type MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call struct {
	*mock.Call
}

// RecvMsg is a helper method to define mock.On call
//   - m interface{}
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) RecvMsg(m interface{}) *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call{Call: _e.mock.On("RecvMsg", m)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call) Run(run func(m interface{})) *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(interface{}))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call) RunAndReturn(run func(interface{}) error) *MockStreamingNodeHandlerService_ConsumeClient_RecvMsg_Call {
	_c.Call.Return(run)
	return _c
}

// Send provides a mock function with given fields: _a0
func (_m *MockStreamingNodeHandlerService_ConsumeClient) Send(_a0 *streamingpb.ConsumeRequest) error {
	ret := _m.Called(_a0)

	if len(ret) == 0 {
		panic("no return value specified for Send")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(*streamingpb.ConsumeRequest) error); ok {
		r0 = rf(_a0)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_Send_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Send'
type MockStreamingNodeHandlerService_ConsumeClient_Send_Call struct {
	*mock.Call
}

// Send is a helper method to define mock.On call
//   - _a0 *streamingpb.ConsumeRequest
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) Send(_a0 interface{}) *MockStreamingNodeHandlerService_ConsumeClient_Send_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_Send_Call{Call: _e.mock.On("Send", _a0)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Send_Call) Run(run func(_a0 *streamingpb.ConsumeRequest)) *MockStreamingNodeHandlerService_ConsumeClient_Send_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*streamingpb.ConsumeRequest))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Send_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeClient_Send_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Send_Call) RunAndReturn(run func(*streamingpb.ConsumeRequest) error) *MockStreamingNodeHandlerService_ConsumeClient_Send_Call {
	_c.Call.Return(run)
	return _c
}

// SendMsg provides a mock function with given fields: m
func (_m *MockStreamingNodeHandlerService_ConsumeClient) SendMsg(m interface{}) error {
	ret := _m.Called(m)

	if len(ret) == 0 {
		panic("no return value specified for SendMsg")
	}

	var r0 error
	if rf, ok := ret.Get(0).(func(interface{}) error); ok {
		r0 = rf(m)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SendMsg'
type MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call struct {
	*mock.Call
}

// SendMsg is a helper method to define mock.On call
//   - m interface{}
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) SendMsg(m interface{}) *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call{Call: _e.mock.On("SendMsg", m)}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call) Run(run func(m interface{})) *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(interface{}))
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call) Return(_a0 error) *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call) RunAndReturn(run func(interface{}) error) *MockStreamingNodeHandlerService_ConsumeClient_SendMsg_Call {
	_c.Call.Return(run)
	return _c
}

// Trailer provides a mock function with given fields:
func (_m *MockStreamingNodeHandlerService_ConsumeClient) Trailer() metadata.MD {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for Trailer")
	}

	var r0 metadata.MD
	if rf, ok := ret.Get(0).(func() metadata.MD); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(metadata.MD)
		}
	}

	return r0
}

// MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Trailer'
type MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call struct {
	*mock.Call
}

// Trailer is a helper method to define mock.On call
func (_e *MockStreamingNodeHandlerService_ConsumeClient_Expecter) Trailer() *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call {
	return &MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call{Call: _e.mock.On("Trailer")}
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call) Run(run func()) *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call) Return(_a0 metadata.MD) *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call) RunAndReturn(run func() metadata.MD) *MockStreamingNodeHandlerService_ConsumeClient_Trailer_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockStreamingNodeHandlerService_ConsumeClient creates a new instance of MockStreamingNodeHandlerService_ConsumeClient. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockStreamingNodeHandlerService_ConsumeClient(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockStreamingNodeHandlerService_ConsumeClient {
	mock := &MockStreamingNodeHandlerService_ConsumeClient{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}